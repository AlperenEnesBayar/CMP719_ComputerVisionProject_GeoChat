"""
Generate paper figures for imgs/ folder.

Figures produced:
  imgs/RSVQA.jpg      - RSVQA LR + HR sample grid
  imgs/AID.jpg        - AID two visually similar classes (viaduct vs bridge)
  imgs/tool_gallery.jpg - 5-panel figure: orig / crop / rotate / contrast / edge
  imgs/qualitative_grid.jpg - Multi-scene GeoAgent step-by-step qualitative grid

Run from project root:
  python scripts/generate_figures.py --model-path ./checkpoints/GeoChat
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Add a white text label in the top-left corner of an image."""
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, font_size + 6], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return img


def hstack(images, gap=4, bg=(255, 255, 255)):
    h = max(i.height for i in images)
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width + gap
    return out


def vstack(images, gap=4, bg=(255, 255, 255)):
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def resize_to_height(img: Image.Image, h: int) -> Image.Image:
    ratio = h / img.height
    return img.resize((int(img.width * ratio), h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Figure 1: RSVQA sample grid
# ---------------------------------------------------------------------------

def make_rsvqa_figure(
    lr_image_folder="datasets/RSVQA_LR/Images_LR",
    hr_image_folder="datasets/RSVQA_HR/Images",
    out_path="imgs/RSVQA.jpg",
    n_samples=3,
    thumb_size=200,
):
    lr_imgs, hr_imgs = [], []

    for folder, store in [(lr_image_folder, lr_imgs), (hr_image_folder, hr_imgs)]:
        files = [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".jpg", ".png"))]
        files.sort()
        for fname in files[:n_samples]:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
            store.append(img)

    if not lr_imgs or not hr_imgs:
        print("RSVQA: missing images, skipping")
        return

    lr_row = hstack([add_label(im, f"LR {i+1}") for i, im in enumerate(lr_imgs)])
    hr_row = hstack([add_label(im, f"HR {i+1}") for i, im in enumerate(hr_imgs)])
    combined = vstack([lr_row, hr_row], gap=6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.save(out_path, quality=92)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: AID – viaduct vs bridge
# ---------------------------------------------------------------------------

def make_aid_figure(
    aid_folder="datasets/AID",
    out_path="imgs/AID.jpg",
    thumb_size=280,
):
    pairs = [("Viaduct", "viaduct"), ("Bridge", "bridge")]
    panels = []
    for label, cls in pairs:
        cls_dir = os.path.join(aid_folder, cls.capitalize())
        if not os.path.isdir(cls_dir):
            # try case-insensitive search
            for d in os.listdir(aid_folder):
                if d.lower() == cls.lower():
                    cls_dir = os.path.join(aid_folder, d)
                    break
        files = sorted(os.listdir(cls_dir))
        img = Image.open(os.path.join(cls_dir, files[0])).convert("RGB")
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        panels.append(add_label(img, label, font_size=20))

    combined = hstack(panels, gap=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.save(out_path, quality=92)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Tool gallery – 5 panels from one image
# ---------------------------------------------------------------------------

def make_tool_gallery(
    source_image=None,
    out_path="imgs/tool_gallery.jpg",
    panel_size=256,
):
    from PIL import ImageEnhance, ImageFilter

    if source_image is None:
        # Use first available demo image
        for fname in ["demo_images/04133.png", "demo_images/04444.png",
                      "demo_images/train_2956_0001.png"]:
            if os.path.exists(fname):
                source_image = fname
                break

    if source_image is None or not os.path.exists(source_image):
        print("Tool gallery: no source image found, skipping")
        return

    orig = Image.open(source_image).convert("RGB")
    orig_sq = orig.resize((panel_size, panel_size), Image.LANCZOS)

    # (a) Original
    pa = add_label(orig_sq.copy(), "(a) Original", 16)

    # (b) Crop – centre 60%
    w, h = orig_sq.size
    cx, cy = int(0.2 * w), int(0.2 * h)
    cropped = orig_sq.crop((cx, cy, w - cx, h - cy)).resize((panel_size, panel_size), Image.LANCZOS)
    pb = add_label(cropped, "(b) Crop", 16)

    # (c) Rotate 90°
    rotated = orig_sq.rotate(-90, expand=False)
    pc = add_label(rotated, "(c) Rotate 90°", 16)

    # (d) Contrast ×2.5
    enhanced = ImageEnhance.Contrast(orig_sq).enhance(2.5)
    pd = add_label(enhanced, "(d) Contrast ×2.5", 16)

    # (e) Edge detection
    edges = orig_sq.convert("RGB").filter(ImageFilter.FIND_EDGES)
    pe = add_label(edges, "(e) Edge Detection", 16)

    row = hstack([pa, pb, pc, pd, pe], gap=6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row.save(out_path, quality=92)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Qualitative multi-scene grid (no model needed – just dataset samples)
# ---------------------------------------------------------------------------

def make_qualitative_grid(
    aid_folder="datasets/AID",
    ucmerced_folder="datasets/UCMerced/UCMerced_LandUse/Images",
    out_path="imgs/qualitative_grid.jpg",
    thumb_size=200,
):
    """3×2 grid of diverse scene types with class labels."""
    scenes = [
        (aid_folder, "Airport"),
        (aid_folder, "Beach"),
        (aid_folder, "Forest"),
        (ucmerced_folder, "agricultural"),
        (ucmerced_folder, "beach"),
        (ucmerced_folder, "buildings"),
    ]

    panels = []
    for base, cls in scenes:
        cls_dir = None
        for d in os.listdir(base):
            if d.lower() == cls.lower():
                cls_dir = os.path.join(base, d)
                break
        if cls_dir is None:
            continue
        files = sorted(os.listdir(cls_dir))
        if not files:
            continue
        img = Image.open(os.path.join(cls_dir, files[0])).convert("RGB")
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        panels.append(add_label(img, cls.capitalize(), 16))

    if not panels:
        print("Qualitative grid: no images found")
        return

    rows = [hstack(panels[i:i+3], gap=6) for i in range(0, len(panels), 3)]
    combined = vstack(rows, gap=6)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.save(out_path, quality=92)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
                        help="GeoChat model path (not needed for static figures)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=== Generating paper figures ===")
    make_rsvqa_figure()
    make_aid_figure()
    make_tool_gallery()
    make_qualitative_grid()
    print("=== Done ===")
