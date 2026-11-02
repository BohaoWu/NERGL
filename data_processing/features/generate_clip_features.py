# coding:utf-8
"""Generate VinVL-compatible region features using YOLO boxes + CLIP ViT-L/14.

Uses YOLO bounding boxes (from pre-generated .npz) and extracts CLIP visual
features for each region crop. CLIP ViT-L/14 outputs 768-dim features,
zero-padded to 2048 to match VinVL format.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os
import argparse
import numpy as np
import torch
from PIL import Image
import open_clip


def build_clip_extractor(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device
    )
    model.eval()
    return model, preprocess


def extract_clip_features(image, boxes, model, preprocess, device, target_dim=2048):
    w, h = image.size
    features = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            features.append(np.zeros(target_dim, dtype=np.float32))
            continue
        crop = image.crop((x1, y1, x2, y2))
        tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
        feat_np = feat.cpu().numpy().flatten().astype(np.float64)
        # Zero-pad from CLIP dim (768) to target_dim (2048)
        if len(feat_np) < target_dim:
            padded = np.zeros(target_dim, dtype=np.float64)
            padded[:len(feat_np)] = feat_np
            feat_np = padded
        features.append(feat_np)
    return np.array(features, dtype=np.float64)


def process_images(image_dir, yolo_dir, output_dir, device_id=1):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print("Loading CLIP ViT-L/14...")
    model, preprocess = build_clip_extractor(device)

    npz_files = sorted([f for f in os.listdir(yolo_dir) if f.endswith('.npz')])
    print(f"Found {len(npz_files)} YOLO feature files")

    for idx, npz_file in enumerate(npz_files):
        out_path = os.path.join(output_dir, npz_file)
        if os.path.exists(out_path):
            continue

        img_name = npz_file.replace('.npz', '')  # e.g. "image.jpg"
        img_path = os.path.join(image_dir, img_name)

        yolo_data = np.load(os.path.join(yolo_dir, npz_file), allow_pickle=True)
        num_boxes = int(yolo_data['num_boxes'])
        boxes = yolo_data['bounding_boxes']
        scores = yolo_data['scores']

        if num_boxes == 0 or not os.path.exists(img_path):
            np.savez(out_path,
                     num_boxes=num_boxes,
                     bounding_boxes=boxes,
                     box_features=np.zeros((num_boxes, 2048), dtype=np.float64) if num_boxes > 0
                                  else np.zeros((0, 2048), dtype=np.float64),
                     scores=scores)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{idx}] Error loading {img_name}: {e}")
            np.savez(out_path, num_boxes=num_boxes, bounding_boxes=boxes,
                     box_features=np.zeros((num_boxes, 2048), dtype=np.float64), scores=scores)
            continue

        box_features = extract_clip_features(image, boxes, model, preprocess, device)

        np.savez(out_path,
                 num_boxes=num_boxes,
                 bounding_boxes=boxes,
                 box_features=box_features,
                 scores=scores)

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(npz_files)}")

    print(f"Done. CLIP features saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP ViT-L/14 region features from YOLO boxes")
    parser.add_argument("--image_dir", default=cfg.IMAGE_DIR + "/")
    parser.add_argument("--yolo_dir", default=os.path.join(cfg.DATA_DIR, "Ukiyoe1000_Yolo/"))
    parser.add_argument("--output_dir", default=os.path.join(cfg.DATA_DIR, "Ukiyoe1000_CLIP/"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    process_images(args.image_dir, args.yolo_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
