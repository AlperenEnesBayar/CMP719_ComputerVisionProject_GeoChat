"""Iterative reasoning agent for GeoChat.

GeoAgent wraps the existing GeoChat model stack with a small control loop:
planner -> optional tool call -> memory update -> repeat -> final answer.

"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import torch
from PIL import Image

from geochat.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from geochat.conversation import SeparatorStyle, conv_templates
from geochat.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from geochat.model.builder import load_pretrained_model


ACTION_SCHEMA = '{"action": "tool|answer|stop", "tool": "crop", "arguments": {}, "final": "optional final answer"}'


@dataclass
class GeoAgentStep:
    step_index: int
    planner_output: str
    action: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None
    final_answer: Optional[str] = None


@dataclass
class GeoAgentMemory:
    query: str = ""
    observations: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    steps: List[GeoAgentStep] = field(default_factory=list)

    def summary(self, max_items: int = 6) -> str:
        lines: List[str] = []
        if self.query:
            lines.append(f"Task: {self.query}")
        if self.decisions:
            lines.append("Decisions:")
            for item in self.decisions[-max_items:]:
                lines.append(f"- {item}")
        if self.observations:
            lines.append("Observations:")
            for item in self.observations[-max_items:]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()


@dataclass
class GeoAgentResult:
    answer: str
    memory: GeoAgentMemory
    raw_planner_trace: List[GeoAgentStep]


@dataclass
class GeoAgentToolResult:
    tool_name: str
    content: str
    image: Optional[Image.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeoAgentTool(Protocol):
    name: str
    description: str

    def run(self, *, image: Optional[Image.Image], memory: GeoAgentMemory, arguments: Dict[str, Any]) -> GeoAgentToolResult:
        ...


@dataclass
class CallableTool:
    name: str
    description: str
    fn: Callable[[Optional[Image.Image], GeoAgentMemory, Dict[str, Any]], GeoAgentToolResult]

    def run(self, *, image: Optional[Image.Image], memory: GeoAgentMemory, arguments: Dict[str, Any]) -> GeoAgentToolResult:
        return self.fn(image, memory, arguments)


@dataclass
class CropTool:
    name: str = "crop"
    description: str = (
        "Crop a rectangular region from the current image. "
        "Arguments: bbox=[x1, y1, x2, y2] in pixels or normalized coordinates."
    )

    def run(self, *, image: Optional[Image.Image], memory: GeoAgentMemory, arguments: Dict[str, Any]) -> GeoAgentToolResult:
        if image is None:
            raise ValueError("crop tool requires an image")

        bbox = arguments.get("bbox") or arguments.get("box") or arguments.get("region")
        if bbox is None:
            raise ValueError("crop tool requires a bbox argument")
        if len(bbox) != 4:
            raise ValueError("bbox must contain four values: [x1, y1, x2, y2]")

        width, height = image.size
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height

        left = int(max(0, min(x1, x2)))
        top = int(max(0, min(y1, y2)))
        right = int(min(width, max(x1, x2)))
        bottom = int(min(height, max(y1, y2)))

        if right <= left or bottom <= top:
            raise ValueError("crop tool produced an empty region")

        cropped = image.crop((left, top, right, bottom))
        content = f"Cropped region: [{left}, {top}, {right}, {bottom}] from image size {width}x{height}."
        return GeoAgentToolResult(tool_name=self.name, content=content, image=cropped, metadata={"bbox": [left, top, right, bottom]})


def _infer_conv_mode(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate_texts: List[str] = []
    stripped = text.strip()
    if stripped:
        candidate_texts.append(stripped)

    code_block_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if code_block_match:
        candidate_texts.insert(0, code_block_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidate_texts.insert(0, brace_match.group(0).strip())

    for candidate in candidate_texts:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class GeoAgent:
    """Iterative reasoning wrapper around a loaded GeoChat model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        image_processor: Any,
        *,
        conv_mode: Optional[str] = None,
        device: str = "cuda",
        tools: Optional[Sequence[GeoAgentTool]] = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.conv_mode = conv_mode or _infer_conv_mode(getattr(model.config, "model_type", model.__class__.__name__))
        self.tools: Dict[str, GeoAgentTool] = {}
        for tool in tools or []:
            self.register_tool(tool)
        self.memory = GeoAgentMemory()
        self.base_image: Optional[Image.Image] = None
        self.current_image: Optional[Image.Image] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_base: Optional[str] = None,
        *,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        conv_mode: Optional[str] = None,
        tools: Optional[Sequence[GeoAgentTool]] = None,
    ) -> "GeoAgent":
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=device,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            conv_mode=conv_mode or _infer_conv_mode(model_name),
            device=device,
            tools=tools,
        )

    def register_tool(self, tool: GeoAgentTool) -> None:
        self.tools[tool.name] = tool

    def reset(self, *, image: Optional[Image.Image] = None, query: str = "") -> None:
        self.base_image = image
        self.current_image = image
        self.memory = GeoAgentMemory(query=query)

    def _image_tensor(self, image: Optional[Image.Image]) -> Optional[torch.Tensor]:
        if image is None:
            return None
        tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device, dtype=torch.float16)

    def _build_prompt(self, query: str, planner: bool) -> str:
        conv = conv_templates[self.conv_mode].copy()
        role_user, role_assistant = conv.roles

        image_prefix = None
        if self.current_image is not None:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                image_prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
            else:
                image_prefix = DEFAULT_IMAGE_TOKEN

        memory_text = self.memory.summary()
        if planner:
            control_text = (
                "You are GeoAgent, a remote-sensing reasoning agent. "
                "Choose the next action from answer, tool, or stop. "
                f"Return a single JSON object that follows this schema: {ACTION_SCHEMA}."
            )
            parts = [control_text]
            if memory_text:
                parts.append(f"Memory:\n{memory_text}")
            parts.append(f"Query:\n{query}")
        else:
            parts = [
                "Use the reasoning trace and tool observations below to answer the user directly and concisely.",
            ]
            if memory_text:
                parts.append(f"Memory:\n{memory_text}")
            parts.append(f"Query:\n{query}")

        if image_prefix is not None:
            parts.insert(0, image_prefix)

        conv.append_message(role_user, "\n\n".join(parts))
        conv.append_message(role_assistant, None)
        return conv.get_prompt()

    def _generate(self, prompt: str, image: Optional[Image.Image], *, max_new_tokens: int, temperature: float, do_sample: bool) -> str:
        image_tensor = self._image_tensor(image)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        stop_str = conv_templates[self.conv_mode].sep
        if conv_templates[self.conv_mode].sep_style == SeparatorStyle.TWO:
            stop_str = conv_templates[self.conv_mode].sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        generation_kwargs = dict(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)

        return self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _parse_action(self, raw_text: str) -> GeoAgentStep:
        parsed = _extract_json_object(raw_text)
        if parsed is None:
            return GeoAgentStep(
                step_index=len(self.memory.steps),
                planner_output=raw_text,
                action="answer",
                final_answer=raw_text.strip(),
            )

        action = str(parsed.get("action", "answer")).strip().lower()
        tool_name = parsed.get("tool")
        arguments = parsed.get("arguments") or {}
        final_answer = parsed.get("final")
        if action not in {"tool", "answer", "stop"}:
            action = "answer"
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return GeoAgentStep(
            step_index=len(self.memory.steps),
            planner_output=raw_text,
            action=action,
            tool_name=str(tool_name) if tool_name is not None else None,
            arguments=arguments,
            final_answer=str(final_answer).strip() if final_answer is not None else None,
        )

    def _call_tool(self, step: GeoAgentStep) -> GeoAgentToolResult:
        if not step.tool_name:
            raise ValueError("tool call missing tool name")
        tool = self.tools.get(step.tool_name)
        if tool is None:
            raise ValueError(f"tool '{step.tool_name}' is not registered")
        return tool.run(image=self.current_image, memory=self.memory, arguments=step.arguments)

    def run(
        self,
        query: str,
        *,
        image: Optional[Image.Image] = None,
        max_steps: int = 3,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        final_max_new_tokens: int = 256,
    ) -> GeoAgentResult:
        self.reset(image=image, query=query)

        final_answer: Optional[str] = None
        for _ in range(max_steps):
            prompt = self._build_prompt(query, planner=True)
            raw_output = self._generate(prompt, self.current_image, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
            step = self._parse_action(raw_output)
            self.memory.steps.append(step)
            self.memory.decisions.append(f"step {step.step_index}: {step.action}" + (f" -> {step.tool_name}" if step.tool_name else ""))

            if step.action in {"answer", "stop"}:
                final_answer = step.final_answer or raw_output.strip()
                break

            if step.action == "tool":
                try:
                    tool_result = self._call_tool(step)
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    observation = f"Tool error: {exc}"
                    self.memory.observations.append(observation)
                    step.observation = observation
                    continue

                self.memory.observations.append(tool_result.content)
                step.observation = tool_result.content
                if tool_result.image is not None:
                    self.current_image = tool_result.image
                continue

            final_answer = raw_output.strip()
            break

        if final_answer is None:
            final_prompt = self._build_prompt(query, planner=False)
            final_answer = self._generate(final_prompt, self.current_image, max_new_tokens=final_max_new_tokens, temperature=0.0, do_sample=False)

        return GeoAgentResult(answer=final_answer.strip(), memory=copy.deepcopy(self.memory), raw_planner_trace=copy.deepcopy(self.memory.steps))


def build_default_tools() -> List[GeoAgentTool]:
    return [CropTool()]
