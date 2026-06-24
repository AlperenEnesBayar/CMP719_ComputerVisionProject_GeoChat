"""
Qualitative inference demo for paper appendix.

Runs each image through:
  1. Base GeoChat (single pass, no tools)
  2. GeoChat FT + GeoAgent (confidence-gated, logs every tool step)

Saves per-image JSON traces + preprocessed image variants for the appendix.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images


# ---------------------------------------------------------------------------
# Question / answer pairs — one entry per image
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "image":    "image.png",
        "question": "What is the primary land use category of this remote sensing image? "
                    "Is the scene agricultural, commercial, residential, or industrial?",
        "ground_truth": "agricultural / farmland",
        "note": "Farmland with circular roundabout and orchard rows",
    },
    {
        "image":    "image2.png",
        "question": "How many sports fields are visible in this image? "
                    "Describe their type and the surrounding environment.",
        "ground_truth": "Two football pitches next to a body of water",
        "note": "Two football pitches adjacent to a river/lake",
    },
    {
        "image":    "image3.png",
        "question": "What type of urban area is shown? Classify the land use of the "
                    "central building complex visible in the image.",
        "ground_truth": "commercial / institutional urban area with large office buildings",
        "note": "Dense commercial downtown with large office complex",
    },
    {
        "image":    "image4.png",
        "question": "What type of residential environment is depicted? "
                    "Can you identify any notable structure in the scene?",
        "ground_truth": "medium-density residential area with a dome structure (mosque)",
        "note": "Residential buildings and a circular dome (mosque)",
    },
]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    tool: str
    params: Dict[str, Any]
    description: str

@dataclass
class ToolResult:
    tool: str
    saved_path: str
    answer_after: str
    answer_changed: bool

@dataclass
class AgentStep:
    step: int
    phase: str          # "initial" | "tool_apply" | "requery" | "final"
    tool_call: Optional[ToolCall]
    tool_result: Optional[ToolResult]
    answer: str
    reasoning: str

@dataclass
class ImageTrace:
    image: str
    question: str
    ground_truth: str
    note: str
    # Base model
    base_answer: str
    base_latency_ms: float
    # Agent
    agent_steps: List[AgentStep] = field(default_factory=list)
    agent_final_answer: str = ""
    agent_latency_ms: float = 0.0
    confidence_gate_triggered: bool = False
    tools_invoked: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: model forward pass
# ---------------------------------------------------------------------------
def _preprocess(image: Image.Image, image_processor, model_config) -> torch.Tensor:
    tensor = process_images([image], image_processor, model_config)
    if isinstance(tensor, list):
        tensor = tensor[0].unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.half().cuda()


def query_model(model, tokenizer, image_processor,
                image: Image.Image, question: str, conv_mode: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                 .unsqueeze(0).cuda())
    image_tensor = _preprocess(image, image_processor, model.config)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128,
            use_cache=True,
        )
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()
    return output


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def tool_enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def tool_detect_edges(image: Image.Image) -> Image.Image:
    """Canny edge detection returned as an RGB PIL image."""
    gray = np.array(image.convert("L"))
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def tool_crop_center(image: Image.Image, ratio: float = 0.7) -> Image.Image:
    """Crop the central fraction of the image."""
    w, h = image.size
    left   = int(w * (1 - ratio) / 2)
    top    = int(h * (1 - ratio) / 2)
    right  = int(w * (1 + ratio) / 2)
    bottom = int(h * (1 + ratio) / 2)
    return image.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)


def tool_rotate(image: Image.Image, angle: float = 90.0) -> Image.Image:
    return image.rotate(angle, expand=False)


TOOLS = {
    "enhance_contrast": (tool_enhance_contrast, {"factor": 1.5},
                         "Enhance contrast (factor=1.5) to improve texture visibility"),
    "detect_edges":     (tool_detect_edges,      {},
                         "Canny edge detection to highlight structural boundaries"),
    "crop_center":      (tool_crop_center,       {"ratio": 0.7},
                         "Crop central 70% region to focus on the dominant feature"),
    "rotate_90":        (tool_rotate,            {"angle": 90.0},
                         "Rotate image 90° to normalise orientation"),
}


# ---------------------------------------------------------------------------
# Confidence gate: does the answer contain any recognisable semantic content?
# ---------------------------------------------------------------------------
UNINFORMATIVE = {"i", "a", "the", "is", "it", "this", "an", "of", "to", "in",
                 "no", "not", "cannot", "sorry", "unable", "unclear"}

def _is_uninformative(answer: str) -> bool:
    words = set(answer.lower().replace(",", " ").replace(".", " ").split())
    meaningful = words - UNINFORMATIVE
    return len(meaningful) < 3 or len(answer.split()) < 4


# ---------------------------------------------------------------------------
# GeoAgent runner — logs every step
# ---------------------------------------------------------------------------
def run_geoagent(model, tokenizer, image_processor, image_path: str,
                 question: str, conv_mode: str, out_dir: str,
                 image_stem: str) -> tuple[str, List[AgentStep], bool, List[str]]:
    """
    Multi-step confidence-gated agent.

    Returns (final_answer, steps, gate_triggered, tools_invoked).
    """
    image_orig = Image.open(image_path).convert("RGB")
    steps: List[AgentStep] = []
    tools_invoked: List[str] = []
    gate_triggered = False

    # Step 1: initial query on original image
    initial_answer = query_model(model, tokenizer, image_processor,
                                 image_orig, question, conv_mode)
    steps.append(AgentStep(
        step=1,
        phase="initial",
        tool_call=None,
        tool_result=None,
        answer=initial_answer,
        reasoning="Direct inference on the original image.",
    ))

    # Confidence gate: if answer looks weak, invoke tools
    gate_triggered = _is_uninformative(initial_answer)
    current_best = initial_answer

    if gate_triggered:
        tool_sequence = ["enhance_contrast", "detect_edges", "crop_center"]
    else:
        # Always demonstrate at least one tool for the paper even when confident
        tool_sequence = ["enhance_contrast"]

    for tool_name in tool_sequence:
        fn, params, desc = TOOLS[tool_name]
        processed = fn(image_orig, **params)

        # Save tool output image
        tool_img_path = os.path.join(out_dir, f"{image_stem}_tool_{tool_name}.png")
        processed.save(tool_img_path)

        # Re-query model on processed image
        new_answer = query_model(model, tokenizer, image_processor,
                                 processed, question, conv_mode)

        changed = new_answer.strip().lower() != current_best.strip().lower()
        tools_invoked.append(tool_name)

        steps.append(AgentStep(
            step=len(steps) + 1,
            phase="tool_apply",
            tool_call=ToolCall(tool=tool_name, params=params, description=desc),
            tool_result=ToolResult(
                tool=tool_name,
                saved_path=tool_img_path,
                answer_after=new_answer,
                answer_changed=changed,
            ),
            answer=new_answer,
            reasoning=(
                f"Confidence gate {'triggered' if gate_triggered else 'not triggered'}; "
                f"applied {tool_name} to verify/improve initial answer."
            ),
        ))

        if gate_triggered:
            # If gate triggered: take the first tool result that looks informative
            if not _is_uninformative(new_answer):
                current_best = new_answer
                break
        else:
            # If gate not triggered: use the original answer (tool is just for demo)
            current_best = initial_answer

    # Final answer step
    steps.append(AgentStep(
        step=len(steps) + 1,
        phase="final",
        tool_call=None,
        tool_result=None,
        answer=current_best,
        reasoning=(
            "Final answer selected from "
            + ("tool-enhanced inference." if gate_triggered else "initial inference (tool applied for demonstration).")
        ),
    ))

    return current_best, steps, gate_triggered, tools_invoked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    disable_torch_init()
    os.makedirs(args.output_dir, exist_ok=True)

    traces: List[ImageTrace] = [
        ImageTrace(
            image=q["image"],
            question=q["question"],
            ground_truth=q["ground_truth"],
            note=q["note"],
            base_answer="",
            base_latency_ms=0.0,
        )
        for q in QUERIES
    ]

    # --- Pass 1: Base GeoChat (all images) ---
    print("\n=== Loading Base GeoChat ===")
    base_name = get_model_name_from_path(args.base_model_path)
    base_tok, base_model, base_proc, _ = load_pretrained_model(
        args.base_model_path, None, base_name)
    base_model.eval()

    print("\n=== Base GeoChat — running all images ===")
    for i, q in enumerate(QUERIES):
        image_path = os.path.join(args.image_dir, q["image"])
        print(f"\n  [{q['image']}]  {q['note']}")
        t0 = time.time()
        base_answer = query_model(base_model, base_tok, base_proc,
                                  Image.open(image_path).convert("RGB"),
                                  q["question"], args.conv_mode)
        traces[i].base_answer = base_answer
        traces[i].base_latency_ms = (time.time() - t0) * 1000
        print(f"  → {base_answer}")

    # Free base model from GPU before loading FT model
    print("\n=== Unloading base model ===")
    del base_model, base_tok, base_proc
    torch.cuda.empty_cache()

    # --- Load FT + GeoAgent model ---
    print("\n=== Loading GeoChat FT (UCMerced) ===")
    ft_name = get_model_name_from_path(args.ft_model_path)
    ft_tok, ft_model, ft_proc, _ = load_pretrained_model(
        args.ft_model_path, args.base_model_path, ft_name)
    ft_model.eval()

    # --- Pass 2: FT + GeoAgent (all images) ---
    print("\n=== GeoChat FT + GeoAgent — running all images ===")
    for i, q in enumerate(QUERIES):
        image_path = os.path.join(args.image_dir, q["image"])
        stem = os.path.splitext(q["image"])[0]
        print(f"\n{'='*60}")
        print(f"Image : {q['image']}  ({q['note']})")
        print(f"Q     : {q['question']}")

        t0 = time.time()
        final_answer, steps, gate_triggered, tools_invoked = run_geoagent(
            ft_model, ft_tok, ft_proc,
            image_path, q["question"], args.conv_mode,
            args.output_dir, stem,
        )
        traces[i].agent_final_answer = final_answer
        traces[i].agent_latency_ms = (time.time() - t0) * 1000
        traces[i].agent_steps = steps
        traces[i].confidence_gate_triggered = gate_triggered
        traces[i].tools_invoked = tools_invoked

        print(f"\n[GeoChat FT + GeoAgent]")
        for s in steps:
            tag = f"  Step {s.step} [{s.phase}]"
            if s.tool_call:
                print(f"{tag}  tool={s.tool_call.tool}  ({s.tool_call.description})")
            else:
                print(f"{tag}")
            print(f"    answer  : {s.answer}")
            print(f"    reason  : {s.reasoning}")
            if s.tool_result:
                changed = "CHANGED" if s.tool_result.answer_changed else "same"
                print(f"    result  : {changed} → saved {s.tool_result.saved_path}")
        print(f"\n  Confidence gate triggered : {gate_triggered}")
        print(f"  Tools invoked             : {tools_invoked}")
        print(f"  Final answer              : {final_answer}")

    # --- Save full JSON trace ---
    out_json = os.path.join(args.output_dir, "inference_traces.json")

    def _serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(out_json, "w") as f:
        json.dump([asdict(t) for t in traces], f, indent=2)
    print(f"\n\nTraces saved → {out_json}")

    # --- Print summary table ---
    print("\n" + "="*60)
    print(f"{'Image':<14} {'Base Answer (truncated)':<35} {'Agent Answer (truncated)':<35} {'Gate'}")
    print("-"*60)
    for t in traces:
        base_short  = (t.base_answer[:33] + "..") if len(t.base_answer) > 35 else t.base_answer
        agent_short = (t.agent_final_answer[:33] + "..") if len(t.agent_final_answer) > 35 else t.agent_final_answer
        gate_str = "YES" if t.confidence_gate_triggered else "no"
        print(f"{t.image:<14} {base_short:<35} {agent_short:<35} {gate_str}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative inference demo for paper appendix")
    parser.add_argument("--base-model-path", default="./checkpoints/GeoChat")
    parser.add_argument("--ft-model-path",   default="./checkpoints/GeoChat-FT-UCMerced")
    parser.add_argument("--image-dir",        default="./inference_imgs")
    parser.add_argument("--output-dir",       default="./results/inference_demo")
    parser.add_argument("--conv-mode",        default="llava_v1")
    args = parser.parse_args()
    main(args)
