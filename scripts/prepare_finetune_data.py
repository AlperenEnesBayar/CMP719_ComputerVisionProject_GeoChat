"""
Create train/test JSONL splits from UCMerced and AID for GeoChat fine-tuning.

Output files (GeoChat conversation format):
  data/finetune/UCMerced_train.json   (1 680 samples, 80 per class)
  data/finetune/UCMerced_test.jsonl   (  420 samples, 20 per class)
  data/finetune/AID_train.json        (5 000 samples, 50% per class)
  data/finetune/AID_test.jsonl        (  all remaining samples)

Run:
  python scripts/prepare_finetune_data.py
"""

import json
import os
import math

UCMERCED_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
]

UCMERCED_DISPLAY = {
    "agricultural": "agricultural", "airplane": "airplane",
    "baseballdiamond": "baseball diamond", "beach": "beach",
    "buildings": "buildings", "chaparral": "chaparral",
    "denseresidential": "dense residential", "forest": "forest",
    "freeway": "freeway", "golfcourse": "golf course", "harbor": "harbor",
    "intersection": "intersection", "mediumresidential": "medium residential",
    "mobilehomepark": "mobile home park", "overpass": "overpass",
    "parkinglot": "parking lot", "river": "river", "runway": "runway",
    "sparseresidential": "sparse residential", "storagetanks": "storage tanks",
    "tenniscourt": "tennis court",
}

UCMERCED_PROMPT = (
    "Classify the given image in one of the following classes. "
    "Classes: " + ", ".join(UCMERCED_DISPLAY.values()) + ". "
    "\nAnswer in one word or a short phrase."
)

AID_PROMPT_TMPL = (
    "Classify the given image in one of the following classes. "
    "Classes: {classes}. "
    "\nAnswer in one word or a short phrase."
)


def make_conversation(image_rel_path: str, prompt: str, answer: str, uid: str) -> dict:
    return {
        "id": uid,
        "image": image_rel_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def prepare_ucmerced(
    image_root="datasets/UCMerced/UCMerced_LandUse/Images",
    out_dir="data/finetune",
    train_per_class=80,
):
    os.makedirs(out_dir, exist_ok=True)
    train_data, test_data = [], []

    for cls in UCMERCED_CLASSES:
        cls_dir = os.path.join(image_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARNING] UCMerced class folder not found: {cls_dir}")
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".tif"))
        train_files = files[:train_per_class]
        test_files = files[train_per_class:]
        label = UCMERCED_DISPLAY[cls]
        for i, fname in enumerate(train_files):
            train_data.append(make_conversation(
                f"{cls}/{fname}", UCMERCED_PROMPT, label, f"ucm_train_{cls}_{i}"
            ))
        for i, fname in enumerate(test_files):
            test_data.append({
                "question_id": f"{cls}/{fname}",
                "image": f"{cls}/{fname}",
                "text": UCMERCED_PROMPT,
                "ground_truth": label,
            })

    with open(os.path.join(out_dir, "UCMerced_train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(out_dir, "UCMerced_test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"UCMerced: {len(train_data)} train / {len(test_data)} test")


def prepare_aid(
    image_root="datasets/AID",
    out_dir="data/finetune",
    train_ratio=0.5,
):
    os.makedirs(out_dir, exist_ok=True)
    train_data, test_data = [], []

    classes = sorted(d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d)))
    display_names = [c for c in classes]
    aid_prompt = AID_PROMPT_TMPL.format(classes=", ".join(display_names))

    for cls in classes:
        cls_dir = os.path.join(image_root, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".tif")))
        n_train = math.ceil(len(files) * train_ratio)
        train_files = files[:n_train]
        test_files = files[n_train:]
        label = cls
        for i, fname in enumerate(train_files):
            train_data.append(make_conversation(
                f"{cls}/{fname}", aid_prompt, label, f"aid_train_{cls}_{i}"
            ))
        for i, fname in enumerate(test_files):
            test_data.append({
                "question_id": f"{cls}/{fname}",
                "image": f"{cls}/{fname}",
                "text": aid_prompt,
                "ground_truth": label,
            })

    with open(os.path.join(out_dir, "AID_train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(out_dir, "AID_test.jsonl"), "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"AID: {len(train_data)} train / {len(test_data)} test")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Preparing UCMerced fine-tuning data...")
    prepare_ucmerced()
    print("Preparing AID fine-tuning data...")
    prepare_aid()
    print("Done. Files written to data/finetune/")
