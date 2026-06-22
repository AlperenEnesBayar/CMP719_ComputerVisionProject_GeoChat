"""GeoAgent evaluation script for RSVQA (LR and HR benchmarks).

Ground truth is loaded from the official RSVQA dataset JSON files.
Usage:
    python geochat/eval/eval_geoagent_vqa.py \
        --model-path ./checkpoints/GeoChat \
        --question-file ./data/GeoChat-Bench/lrben.jsonl \
        --answers-file ./results/geoagent_lrben.jsonl \
        --image-folder ./datasets/RSVQA_LR/Images_LR \
        --gt-answers ./datasets/RSVQA_LR/LR_split_test_answers.json
"""

import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

from PIL import Image

from geochat.agent import GeoAgent, build_default_tools
from geochat.utils import disable_torch_init
from geochat.mm_utils import get_model_name_from_path


def load_gt_answers(gt_path: str) -> dict:
    """Build {question_id: answer_str} from RSVQA answer JSON."""
    data = json.load(open(gt_path))
    answers = data.get("answers", [])
    gt = {}
    for a in answers:
        if a.get("active", False):
            gt[a["question_id"]] = a["answer"].strip().lower()
    return gt


def evaluate(answers_file: str, gt: dict) -> float:
    results = [json.loads(l) for l in open(answers_file)]
    per_cat: dict = defaultdict(lambda: [0, 0])  # [correct, total]
    total_correct = 0

    for r in results:
        qid = r["question_id"]
        pred = r["answer"].strip().lower()
        truth = gt.get(qid, "")
        correct = int(pred == truth)
        total_correct += correct
        cat = r.get("category", "unknown")
        per_cat[cat][0] += correct
        per_cat[cat][1] += 1

    n = len(results)
    acc = total_correct / n if n else 0.0
    print(f"\nOverall accuracy: {acc*100:.2f}%  ({total_correct}/{n})")
    for cat, (c, t) in sorted(per_cat.items()):
        print(f"  {cat:<20s}: {c/t*100:.2f}%  ({c}/{t})")
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
    gt = load_gt_answers(os.path.expanduser(args.gt_answers)) if args.gt_answers else {}

    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)

    with open(os.path.expanduser(args.answers_file), "w") as ans_file:
        for q in tqdm(questions, desc="GeoAgent VQA eval"):
            image_path = os.path.join(args.image_folder, q["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                continue
            result = agent.run(q["text"], image=image, max_steps=args.max_steps)
            ans_file.write(
                json.dumps({
                    "question_id": q["question_id"],
                    "image": q["image"],
                    "answer": result.answer,
                    "category": q.get("category", ""),
                    "steps": len(result.raw_planner_trace),
                }) + "\n"
            )
            ans_file.flush()

    if gt:
        evaluate(os.path.expanduser(args.answers_file), gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--gt-answers", type=str, default=None,
                        help="Path to RSVQA *_answers.json ground-truth file")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    main(args)
