"""GeoAgent evaluation script for aerial scene classification (UCMerced, AID)."""

import argparse
import json
import os
from tqdm import tqdm

from PIL import Image

from geochat.agent import GeoAgent, build_default_tools
from geochat.utils import disable_torch_init
from geochat.mm_utils import get_model_name_from_path


def evaluation_metrics(answers_file: str) -> float:
    results = [json.loads(l) for l in open(answers_file)]
    correct = 0
    for r in results:
        gt = r["question_id"].split("/")[0].replace(" ", "").lower()
        pred = r["answer"].replace(" ", "").lower().replace(".", "")
        if gt == pred:
            correct += 1
    acc = correct / len(results) if results else 0.0
    print(f"Correct: {correct} / {len(results)}  |  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return acc


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    agent = GeoAgent.from_pretrained(
        model_path,
        args.model_base,
        conv_mode=args.conv_mode,
        tools=build_default_tools(),
    )

    questions = [json.loads(l) for l in open(os.path.expanduser(args.question_file))]
    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)

    with open(os.path.expanduser(args.answers_file), "w") as ans_file:
        for q in tqdm(questions, desc="GeoAgent scene eval"):
            image_path = os.path.join(args.image_folder, q["image"])
            image = Image.open(image_path).convert("RGB")
            result = agent.run(q["text"], image=image, max_steps=args.max_steps)
            ans_file.write(
                json.dumps({
                    "question_id": q["question_id"],
                    "image": q["image"],
                    "answer": result.answer,
                    "ground_truth": q.get("ground_truth", ""),
                    "steps": len(result.raw_planner_trace),
                }) + "\n"
            )
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
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    main(args)
