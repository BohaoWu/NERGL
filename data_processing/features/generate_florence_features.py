# coding:utf-8
"""Generate VinVL-compatible region features using Florence-2 region proposals + ResNet50."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def build_feature_extractor(device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    modules = list(model.children())[:-1]
    return torch.nn.Sequential(*modules).to(device)


def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_box_features(image, boxes, extractor, transform, device):
    w, h = image.size
    features = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            features.append(np.zeros(2048, dtype=np.float32))
            continue
        crop = image.crop((x1, y1, x2, y2))
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(tensor).squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy().flatten())
    return np.array(features, dtype=np.float64)


def process_images(image_dir, output_dir, max_bbox=18, device_id=0):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print("Loading Florence-2-large...")
    florence = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    ).to(device).eval()
    flo_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )

    print("Loading ResNet50 feature extractor")
    extractor = build_feature_extractor(device)
    transform = get_transform()

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Found {len(image_files)} images in {image_dir}")

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        out_path = os.path.join(output_dir, img_file + ".npz")

        if os.path.exists(out_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{idx}] Error loading {img_file}: {e}")
            continue

        inputs = flo_processor(
            text="<REGION_PROPOSAL>", images=image, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = florence.generate(**inputs, max_new_tokens=1024, num_beams=3)
        result = flo_processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = flo_processor.post_process_generation(
            result, task="<REGION_PROPOSAL>",
            image_size=(image.width, image.height)
        )

        bboxes = parsed.get("<REGION_PROPOSAL>", {}).get("bboxes", [])

        if len(bboxes) > 0:
            boxes = np.array(bboxes, dtype=np.float64)
            num_boxes = min(len(boxes), max_bbox)
            boxes = boxes[:num_boxes]
            scores = np.linspace(1.0, 0.5, num_boxes).astype(np.float64)
        else:
            num_boxes = 0
            boxes = np.zeros((0, 4), dtype=np.float64)
            scores = np.array([], dtype=np.float64)

        if num_boxes > 0:
            box_features = extract_box_features(
                image, boxes, extractor, transform, device
            )
        else:
            box_features = np.zeros((0, 2048), dtype=np.float64)

        np.savez(out_path,
                 num_boxes=num_boxes,
                 bounding_boxes=boxes,
                 box_features=box_features,
                 scores=scores)

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(image_files)}")

    print(f"Done. Features saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=cfg.IMAGE_DIR + "/")
    parser.add_argument("--output_dir", default=os.path.join(cfg.DATA_DIR, "Ukiyoe1000_Florence/"))
    parser.add_argument("--max_bbox", type=int, default=18)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    process_images(args.image_dir, args.output_dir, args.max_bbox, args.device)


if __name__ == "__main__":
    main()
