import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
# coding:utf-8
import os
import re
import sys
import time
import json
import base64
import argparse
from collections import Counter
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import anthropic

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


SYSTEM_PROMPT = "あなたは20年以上の経験を持つ浮世絵の専門家です。"

GMNER_PROMPT = """以下の浮世絵のタイトルと画像が与えられます。画像にはVinVLで検出された候補領域（Region）が番号付きで表示されています。

固有表現の種類は次の四つです。
・地名：日本の土地や場所の名称。
・演目：歌舞伎や浄瑠璃などの作品の題名。
・替名：作品の中の登場人物の役名。演じる役者の芸名とは別のものです。
・役者：歌舞伎役者の芸名。

タスク：
1. タイトルから固有表現を抽出してください。種類は上の四つのいずれかとし、他の語は使わないでください。
2. 各固有表現について、画像のどの領域（Region番号）に対応するか判断してください。対応する領域がない場合は番号の代わりに「なし」と書いてください。
3. 同じ固有表現がタイトルに二回現れる場合は、二行に分けて書いてください。

VinVL検出領域の座標（Region番号: [x1, y1, x2, y2]）：
{box_info}

浮世絵タイトル：{title}

以下の形式で回答してください（各固有表現を1行ずつ、他の文は書かないでください）：
「固有表現 種類 Region番号」
例：
「隅田川 地名 3」
「忠臣蔵 演目 なし」
「おかる 替名 1」
「坂東家橘 役者 0」
"""


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_test_file(test_txt_path):
    samples = []
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cur_img_id = None
    cur_chars = []
    cur_labels = []
    for line in lines:
        line = line.strip()
        if line.startswith("IMGID:"):
            if cur_img_id is not None:
                samples.append({
                    "img_id": cur_img_id,
                    "text": "".join(cur_chars),
                    "labels": cur_labels,
                })
            cur_img_id = line.split("IMGID:")[1]
            cur_chars = []
            cur_labels = []
        elif line:
            parts = line.split("\t")
            if len(parts) == 2:
                cur_chars.append(parts[0])
                cur_labels.append(parts[1])
    if cur_img_id is not None:
        samples.append({
            "img_id": cur_img_id,
            "text": "".join(cur_chars),
            "labels": cur_labels,
        })
    return samples


def extract_gt_entities_from_bio(text, labels):
    entities = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            tag = labels[i][2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{tag}":
                i += 1
            end = i
            entity_text = text[start:end]
            entities.append((entity_text, tag, start, end))
        else:
            i += 1
    return entities


def read_xml_annotation(xml_path):
    if not os.path.exists(xml_path):
        return {}, None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_size = None
    if size is not None:
        img_size = (int(size.find("width").text), int(size.find("height").text))

    entity_boxes = {}
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        box = [
            int(bndbox.find("xmin").text), int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text), int(bndbox.find("ymax").text),
        ]
        if name not in entity_boxes:
            entity_boxes[name] = []
        entity_boxes[name].append(box)
    return entity_boxes, img_size


def load_vinvl_boxes(npz_path, max_bbox=18):
    data = np.load(npz_path, allow_pickle=True)
    num_boxes = int(data['num_boxes'])
    final_num = min(num_boxes, max_bbox)
    boxes = data['bounding_boxes'][:final_num]
    return boxes, final_num


def compute_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def get_gt_region_indices(entity_name, entity_boxes, vinvl_boxes, iou_thresh=0.5):
    if entity_name not in entity_boxes:
        return []
    gt_boxes = entity_boxes[entity_name]
    region_indices = set()
    for gt_box in gt_boxes:
        for idx, vbox in enumerate(vinvl_boxes):
            if compute_iou(gt_box, vbox) >= iou_thresh:
                region_indices.add(idx)
    return sorted(region_indices)


def format_box_info(vinvl_boxes):
    lines = []
    for i, box in enumerate(vinvl_boxes):
        lines.append(f"Region {i}: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
    return "\n".join(lines)


def parse_llm_gmner_response(response, text, num_boxes):
    entities = []
    bracket_matches = re.findall(r'「(.*?)」', response)
    if bracket_matches:
        lines = bracket_matches
    else:
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    for m in lines:
        m = m.strip().rstrip("  ")
        parts = m.split()
        if len(parts) < 2:
            continue
        if len(parts) >= 3:
            entity_text = parts[0]
            label = parts[1]
            region_str = parts[2]
        else:
            entity_text = parts[0]
            label = parts[1]
            region_str = "なし"

        label_map = {
            "地名": "地名", "役者": "役者", "役者名": "役者",
            "替名": "替名", "演目": "演目", "力士名": "力士名",
            "座本": "座本", "絵師": "絵師", "シリーズ名": "シリーズ名",
            "屋号": "屋号", "武将": "武将",
        }
        label = label_map.get(label, label)

        start = text.find(entity_text)
        if start == -1:
            continue
        end = start + len(entity_text)

        if region_str == "なし" or region_str == "none":
            region_indices = [num_boxes]
        else:
            try:
                region_indices = [int(r) for r in region_str.replace(",", " ").split() if r.isdigit()]
                if not region_indices:
                    region_indices = [num_boxes]
            except ValueError:
                region_indices = [num_boxes]

        entities.append({
            "text": entity_text,
            "label": label,
            "span": (start, end),
            "regions": region_indices,
        })
    return entities


# The model name sent to the API is an alias that the provider may repoint to a
# newer snapshot, so every response records the snapshot that actually served it.
SNAPSHOTS = Counter()

REASONING_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.5", "gpt-6", "o3", "o4-mini"}


def call_gpt(client, image_path, title, box_info, model="gpt-4.1"):
    """Every model receives the same text, the same image, and the same order.

    The persona sits inside the user text rather than in a system message,
    because reasoning models were previously sent no system message at all,
    which made their input differ from the other models'. Sampling is turned
    off where the model allows it, so a run can be reproduced; a model that
    rejects the setting is retried without it and that is reported.
    """
    img_b64 = encode_image_base64(image_path)
    prompt = SYSTEM_PROMPT + "\n\n" + GMNER_PROMPT.format(box_info=box_info, title=title)
    is_reasoning = any(model.startswith(r) for r in REASONING_MODELS)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    # Reasoning models spend the completion budget on hidden reasoning first; at 3000
    # tokens GPT-6 Luna ran out on 36 of 1079 prints and returned no answer at all.
    budget = {"max_completion_tokens": 8000} if is_reasoning else {"max_tokens": 3000}
    for temp in ({"temperature": 0}, {}):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **budget, **temp)
            SNAPSHOTS[response.model] += 1
            return response.choices[0].message.content or ""
        except Exception as e:
            if temp and "temperature" in str(e):
                print(f"  {model}: temperature unsupported, sampling left at default")
                continue
            print(f"  GPT error: {e}")
            time.sleep(1)
            return ""
    return ""


def call_claude(client, image_path, title, box_info, model="claude-sonnet-4-5-20250514"):
    img_b64 = encode_image_base64(image_path)
    prompt = SYSTEM_PROMPT + "\n\n" + GMNER_PROMPT.format(box_info=box_info, title=title)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=0,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        )
        SNAPSHOTS[message.model] += 1
        return message.content[0].text
    except Exception as e:
        print(f"  Claude error: {e}")
        time.sleep(1)
        return ""


def compute_gmner_metrics(all_preds, all_gts):
    tp = fp = fn = 0
    ner_tp = ner_fp = ner_fn = 0
    grounding_correct = grounding_total = 0

    for preds, gts in zip(all_preds, all_gts):
        pred_ner = {(p["text"], p["label"]) for p in preds}
        gt_ner = {(g["text"], g["label"]) for g in gts}
        ner_tp += len(pred_ner & gt_ner)
        ner_fp += len(pred_ner - gt_ner)
        ner_fn += len(gt_ner - pred_ner)

        gt_map = {(g["text"], g["label"]): g["regions"] for g in gts}
        for p in preds:
            key = (p["text"], p["label"])
            if key in gt_map:
                gt_regions = set(gt_map[key])
                pred_regions = set(p["regions"])
                if len(gt_regions & pred_regions) > 0:
                    tp += 1
                    grounding_correct += 1
                else:
                    fp += 1
                grounding_total += 1
            else:
                fp += 1
        for g in gts:
            key = (g["text"], g["label"])
            pred_keys = {(p["text"], p["label"]) for p in preds}
            if key not in pred_keys:
                fn += 1

    ner_p = ner_tp / (ner_tp + ner_fp) if (ner_tp + ner_fp) > 0 else 0
    ner_r = ner_tp / (ner_tp + ner_fn) if (ner_tp + ner_fn) > 0 else 0
    ner_f1 = 2 * ner_p * ner_r / (ner_p + ner_r) if (ner_p + ner_r) > 0 else 0

    gmner_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    gmner_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    gmner_f1 = 2 * gmner_p * gmner_r / (gmner_p + gmner_r) if (gmner_p + gmner_r) > 0 else 0

    grounding_acc = grounding_correct / grounding_total if grounding_total > 0 else 0

    return {
        "NER": {"P": ner_p, "R": ner_r, "F1": ner_f1},
        "GMNER": {"P": gmner_p, "R": gmner_r, "F1": gmner_f1},
        "Grounding": {"acc": grounding_acc, "correct": grounding_correct, "total": grounding_total},
    }


def estimate_cost(n_samples, model, has_image=True):
    avg_text_tokens = 700
    avg_output_tokens = 80
    avg_image_tokens = 1100 if has_image else 0

    reasoning_tokens_per_sample = {
        "gpt-5.5": 512, "o3": 1024,
        "gpt-6-astra": 512, "gpt-6.1-sol": 512, "gpt-6-luna": 512,
    }
    avg_reasoning = reasoning_tokens_per_sample.get(model, 0)

    total_input = (avg_text_tokens + avg_image_tokens) * n_samples
    total_output = (avg_output_tokens + avg_reasoning) * n_samples

    prices = {
        "gpt-4.1":                    (2.0, 8.0),
        "gpt-4.1-mini":               (0.4, 1.6),
        "gpt-4.1-nano":               (0.1, 0.4),
        "gpt-4o":                     (2.5, 10.0),
        "gpt-4o-mini":                (0.15, 0.6),
        "gpt-5.4":                    (2.0, 8.0),
        "gpt-5.4-mini":               (0.4, 1.6),
        "gpt-5.4-nano":               (0.1, 0.4),
        "gpt-5.5":                    (2.0, 8.0),
        "gpt-6-astra":                (10.0, 50.0),
        "gpt-6.1-sol":                (2.0, 10.0),
        "gpt-6-luna":                 (0.10, 0.40),
        "o3":                         (2.0, 8.0),
        "o4-mini":                    (0.4, 1.6),
        "claude-sonnet-4-5-20250514": (2.0, 10.0),
        "claude-3-haiku-20240307":    (0.25, 1.25),
    }
    ip, op = prices.get(model, (2.0, 8.0))
    cost = total_input * ip / 1e6 + total_output * op / 1e6

    print(f"\n=== Cost Estimate: {model} ===")
    print(f"  Samples:        {n_samples}")
    print(f"  Input tokens:   ~{total_input:,} (text: {avg_text_tokens}, image: {avg_image_tokens})")
    print(f"  Output tokens:  ~{total_output:,}")
    print(f"  Est. cost:      ${cost:.4f}")
    return cost


GPT_MODELS = {
    "gpt-4.1": "gpt-4.1",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.5": "gpt-5.5",
    "gpt-6-astra": "gpt-6-astra",
    "gpt-6.1-sol": "gpt-6.1-sol",
    "gpt-6-luna": "gpt-6-luna",
    "o3": "o3",
}


def run_evaluation(test_txt, image_dir, vinvl_dir, xml_dir, output_dir,
                   model_type="both", dry_run=False, max_bbox=18, concurrency=5):
    samples = parse_test_file(test_txt)
    print(f"Loaded {len(samples)} test samples")

    models = []
    if model_type == "both" or model_type == "gpt":
        for label, model_id in GPT_MODELS.items():
            models.append((label, model_id, "gpt"))
    if model_type == "both" or model_type == "claude":
        models.append(("claude", "claude-sonnet-4-5-20250514", "claude"))
    if model_type in GPT_MODELS:
        models = [(model_type, GPT_MODELS[model_type], "gpt")]

    total_cost = 0
    for name, model, _ in models:
        total_cost += estimate_cost(len(samples), model)
    print(f"\n  Total est. cost: ${total_cost:.4f}")

    if dry_run:
        print("\n[Dry run] Skipping API calls.")
        return

    gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    for name, model, provider in models:
        print(f"\n{'='*50}")
        print(f"Running {name} ({model})")
        print(f"{'='*50}")

        all_preds = [None] * len(samples)
        all_gts = [None] * len(samples)
        result_lines_map = {}
        skipped = set()

        def process_sample(idx, sample):
            img_id = sample["img_id"]
            text = sample["text"]

            image_path = os.path.join(image_dir, img_id + ".jpg")
            npz_path = os.path.join(vinvl_dir, img_id + ".jpg.npz")
            xml_path = os.path.join(xml_dir, img_id + ".xml")

            if not os.path.exists(image_path) or not os.path.exists(npz_path):
                print(f"  [{idx}] Skip {img_id}: missing image or VinVL")
                return idx, None, None, None

            vinvl_boxes, num_boxes = load_vinvl_boxes(npz_path, max_bbox)
            box_info = format_box_info(vinvl_boxes)
            entity_boxes, _ = read_xml_annotation(xml_path)

            gt_bio_entities = extract_gt_entities_from_bio(text, sample["labels"])
            gt_entries = []
            for ent_text, ent_tag, start, end in gt_bio_entities:
                gt_regions = get_gt_region_indices(ent_text, entity_boxes, vinvl_boxes)
                if not gt_regions:
                    gt_regions = [num_boxes]
                gt_entries.append({
                    "text": ent_text,
                    "label": ent_tag,
                    "span": (start, end),
                    "regions": gt_regions,
                })

            clean_title = text.replace("_", " ")
            if provider == "gpt":
                response = call_gpt(gpt_client, image_path, clean_title, box_info, model)
            else:
                response = call_claude(claude_client, image_path, clean_title, box_info, model)

            pred_entries = parse_llm_gmner_response(response, text, num_boxes)

            lines = (
                f"{img_id}\t{text}\n"
                f"  LLM: {response}\n"
                f"  Pred: {[(e['text'], e['regions'], e['label']) for e in pred_entries]}\n"
                f"  GT:   {[(e['text'], e['regions'], e['label']) for e in gt_entries]}\n\n"
            )
            return idx, pred_entries, gt_entries, lines

        done_count = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_sample, idx, s): idx for idx, s in enumerate(samples)}
            for future in as_completed(futures):
                idx, preds, gts, lines = future.result()
                done_count += 1
                if preds is None:
                    skipped.add(idx)
                else:
                    all_preds[idx] = preds
                    all_gts[idx] = gts
                    result_lines_map[idx] = lines
                if done_count % 10 == 0:
                    print(f"  Progress: {done_count}/{len(samples)}")

        all_preds = [p for i, p in enumerate(all_preds) if i not in skipped and p is not None]
        all_gts = [g for i, g in enumerate(all_gts) if i not in skipped and g is not None]
        result_lines = [result_lines_map[i] for i in sorted(result_lines_map.keys())]

        metrics = compute_gmner_metrics(all_preds, all_gts)
        print(f"\n--- {name} Results ---")
        print(f"  NER:       P={metrics['NER']['P']:.4f}  R={metrics['NER']['R']:.4f}  F1={metrics['NER']['F1']:.4f}")
        print(f"  GMNER:     P={metrics['GMNER']['P']:.4f}  R={metrics['GMNER']['R']:.4f}  F1={metrics['GMNER']['F1']:.4f}")
        print(f"  Grounding: Acc={metrics['Grounding']['acc']:.4f} ({metrics['Grounding']['correct']}/{metrics['Grounding']['total']})")

        output_file = os.path.join(output_dir, f"{name}_gmner_result.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(result_lines)
            f.write(f"\n--- Metrics ---\n")
            f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
            f.write(f"\n--- Snapshots ---\n")
            f.write(json.dumps(dict(SNAPSHOTS), indent=2))
        print(f"  Results saved to {output_file}; snapshots {dict(SNAPSHOTS)}")
        SNAPSHOTS.clear()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on GMNER (NER + Grounding)")
    parser.add_argument("--test_txt", default=os.path.join(cfg.TXT_DIR, "test.txt"))
    parser.add_argument("--image_dir", default=cfg.IMAGE_DIR + "/")
    parser.add_argument("--vinvl_dir", default=cfg.VINVL_DIR + "/")
    parser.add_argument("--xml_dir", default=cfg.XML_DIR + "/")
    parser.add_argument("--output_dir", default=cfg.SAVE_DIR + "/")
    parser.add_argument("--model", default="both", choices=["gpt", "claude", "both", "gpt-4.1", "gpt-5.4", "gpt-5.5", "gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "o3"])
    parser.add_argument("--dry_run", action="store_true", help="Only estimate cost")
    parser.add_argument("--max_bbox", type=int, default=18)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    run_evaluation(
        args.test_txt, args.image_dir, args.vinvl_dir, args.xml_dir,
        args.output_dir, args.model, args.dry_run, args.max_bbox, args.concurrency,
    )


if __name__ == "__main__":
    main()
