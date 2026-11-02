# coding:utf-8
"""Grounding DINO region features taken from the model itself.

The ResNet50 variants (generate_gdino_features.py, generate_gdino_prompt_features.py)
keep only the Grounding DINO boxes and re-encode each crop. Here every box keeps the
decoder query state that produced it, a 256-dimensional vector fused with the text
prompt, zero-padded to 2048 to match the VinVL format. The boxes are selected exactly
as in the ResNet50 scripts, and each image is checked against the boxes already on
disk so the two variants differ only in their features.
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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.models.grounding_dino.processing_grounding_dino import center_to_corners_format

PROMPTS = {
    "base": "person . face . text . building . tree . clothing . sign . animal . wave . mountain .",
    "prompt": (
        "person . face . text . building . tree . clothing . sign . animal . wave . mountain . "
        "bridge . river . sword . fan . flower . samurai . kimono . actor . figure . landscape . "
        "castle . boat . horse . cloud . moon . pine . bamboo . snow . pagoda . gate ."
    ),
}


def process_images(image_dir, output_dir, ref_dir, text_prompt, max_bbox=18,
                   box_threshold=0.1, device_id=0, target_dim=2048):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    gdino.eval()

    image_files = sorted(f for f in os.listdir(image_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(f"Found {len(image_files)} images in {image_dir}", flush=True)
    worst, mismatched = 0.0, []

    for idx, img_file in enumerate(image_files):
        out_path = os.path.join(output_dir, img_file + ".npz")
        if os.path.exists(out_path):
            continue
        try:
            image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        except Exception as e:
            print(f"  [{idx}] Error loading {img_file}: {e}", flush=True)
            continue

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gdino(**inputs)

        # Same selection as post_process_grounded_object_detection followed by the
        # descending sort and top-K cut of the ResNet50 scripts, but keeping the
        # query index of every box.
        scores = torch.sigmoid(outputs.logits[0]).max(dim=-1)[0]
        keep = torch.nonzero(scores > box_threshold).squeeze(-1)
        w, h = image.size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
        boxes = center_to_corners_format(outputs.pred_boxes[0])[keep] * scale
        scores_k = scores[keep].cpu().numpy()
        order = np.argsort(-scores_k)[:max_bbox]
        q_idx = keep.cpu().numpy()[order]

        num_boxes = len(q_idx)
        boxes = boxes.cpu().numpy()[order].astype(np.float64)
        feats = np.zeros((num_boxes, target_dim), dtype=np.float64)
        if num_boxes:
            q = outputs.last_hidden_state[0, torch.as_tensor(q_idx, device=device)]
            feats[:, :q.size(-1)] = q.cpu().numpy()

        ref = os.path.join(ref_dir, img_file + ".npz") if ref_dir else None
        if ref and os.path.exists(ref):
            rb = np.load(ref)["bounding_boxes"]
            if rb.shape != boxes.shape:
                mismatched.append(img_file)
            elif len(rb):
                worst = max(worst, float(np.abs(rb - boxes).max()))

        np.savez(out_path, num_boxes=num_boxes, bounding_boxes=boxes,
                 box_features=feats, scores=scores_k[order].astype(np.float64))
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(image_files)}  max box diff {worst:.3f}px  "
                  f"count mismatches {len(mismatched)}", flush=True)

    print(f"Done. max box diff vs {ref_dir}: {worst:.3f}px; "
          f"count mismatches: {len(mismatched)} {mismatched[:10]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--image_dir", default=cfg.IMAGE_DIR + "/")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ref_dir", default=None,
                    help="optional ResNet50 variant whose boxes these must reproduce")
    ap.add_argument("--max_bbox", type=int, default=18)
    ap.add_argument("--box_threshold", type=float, default=0.1)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    process_images(a.image_dir, a.output_dir, a.ref_dir, PROMPTS[a.variant],
                   a.max_bbox, a.box_threshold, a.device)


if __name__ == "__main__":
    main()
