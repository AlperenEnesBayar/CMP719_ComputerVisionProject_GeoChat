"""GeoAgent scene classification eval — confidence-gated tool pipeline.

GeoAgent first asks GeoChat directly (Step 1). If the answer does not
contain any known class name, the agent applies contrast enhancement
(Step 2) and re-queries the model.  This mimics a confidence-gated
agent: tools are invoked only when the initial prediction is unclear.
"""

import argparse
import json
import os

import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images


def _preprocess(image: Image.Image, image_processor, model_config) -> torch.Tensor:
    tensor = process_images([image], image_processor, model_config)
    if isinstance(tensor, list):
        tensor = tensor[0].unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.half().cuda()


def _run_model(model, tokenizer, image_processor, image: Image.Image, question: str, conv_mode: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image_tensor = _preprocess(image, image_processor, model.config)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=64,
            use_cache=True,
        )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    if output.endswith(stop_str):
        output = output[:-len(stop_str)].strip()
    return output


def _extract_classes_from_question(text: str):
    """Pull class list from the benchmark question text (after 'Classes:')."""
    if "Classes:" not in text:
        return []
    after = text.split("Classes:")[1].split("\n")[0].strip()
    return [c.strip().lower().replace(" ", "") for c in after.split(",") if c.strip()]


def evaluation_metrics(answers_file: str) -> float:
    results = [json.loads(l) for l in open(answers_file)]
    correct = 0
    tool_used = 0
    for r in results:
        gt = r["question_id"].split("/")[0].replace(" ", "").lower()
        pred = r["answer"].replace(" ", "").lower().replace(".", "").replace(",", "")
        if gt == pred or gt in pred:
            correct += 1
        if r.get("tool_used") != "none":
            tool_used += 1
    acc = correct / len(results) if results else 0.0
    print(f"Correct: {correct} / {len(results)}  |  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Tool invocations: {tool_used} / {len(results)} ({tool_used/len(results)*100:.1f}%)")
    return acc


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()

    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    known_classes = _extract_classes_from_question(questions[0]["text"]) if questions else []

    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)

    with open(os.path.expanduser(args.answers_file), "w") as ans_file:
        for q in tqdm(questions, desc="GeoAgent scene eval"):
            image_path = os.path.join(args.image_folder, q["image"])
            image = Image.open(image_path).convert("RGB")

            # Step 1: direct classification (same as base eval)
            answer = _run_model(model, tokenizer, image_processor, image, q["text"], args.conv_mode)
            answer_clean = answer.replace(" ", "").lower().replace(".", "").replace(",", "")

            tool_used = "none"

            # Check if answer matches a known class
            if known_classes and not any(cls in answer_clean for cls in known_classes):
                # Step 2: confidence gate triggered — apply contrast enhancement
                enhanced = ImageEnhance.Contrast(image).enhance(args.contrast_factor)
                answer = _run_model(model, tokenizer, image_processor, enhanced, q["text"], args.conv_mode)
                tool_used = "enhance_contrast"

            ans_file.write(json.dumps({
                "question_id": q["question_id"],
                "image": q["image"],
                "answer": answer,
                "ground_truth": q.get("ground_truth", ""),
                "tool_used": tool_used,
            }) + "\n")
            ans_file.flush()

    evaluation_metrics(os.path.expanduser(args.answers_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--contrast-factor", type=float, default=1.5,
                        help="Contrast factor applied when confidence gate triggers")
    args = parser.parse_args()
    main(args)
