"""Fetch everything the experiments need that is not in the repository.

  python3 script/download.py                 # dataset, images, knowledge cache and BART
  python3 script/download.py --proposers     # also the region features of the proposer comparison
  python3 script/download.py --model         # also the trained NERGL weights

The annotations, splits, region features and knowledge cache come from the
dataset repository on HuggingFace. The prints themselves are not redistributed:
images.csv there lists each print's public URL at the Art Research Center,
Ritsumeikan University, with its SHA-256, and the images are fetched from that
source and checked against the hash. Please follow the ARC terms of use.

Files land where data_processing/config.py says (DATA_DIR, SAVE_DIR, MODEL_DIR),
so a path changed in .env is followed here too. Files already present and
correct are not fetched again, so an interrupted run can simply be repeated.
"""
import os
import csv
import sys
import time
import shutil
import zipfile
import hashlib
import argparse
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_processing import config as cfg     # every path of the project
from huggingface_hub import snapshot_download, hf_hub_download

BART_REPO = "ku-nlp/bart-large-japanese"
DATA_PARTS = ["txt/*", "xml/*", "cv/*", "images.csv"]
# region features, one zip per proposer, unpacked into DATA_DIR/<name>/
FEATURES = ["Ukiyoe1000_VinVL"]
PROPOSER_FEATURES = ["Ukiyoe1000_GDino_VL", "Ukiyoe1000_GDino_Prompt_VL", "Ukiyoe1000_Florence",
                     "Ukiyoe1000_RTDETR", "Ukiyoe1000_Yolo", "Ukiyoe1000_CLIP"]
KNOWLEDGE = "knowledge/" + os.path.basename(cfg.KNOWLEDGE_FILE)
MODEL_FILE = "nergl_vinvl_f0_s16_bestgmner.bin"
HEADERS = {"User-Agent": "NERGL-dataset-fetch/1.0 (research use)", "Referer": "http://www.arc.ritsumei.ac.jp/"}
TOKEN = os.environ.get("HF_TOKEN") or None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_features(name):
    dest = os.path.join(cfg.DATA_DIR, name)
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 1078:
        return
    print("  %s -> %s" % (name, dest))
    z = hf_hub_download(cfg.HF_DATA_REPO, "features/%s.zip" % name, repo_type="dataset", token=TOKEN)
    with zipfile.ZipFile(z) as f:
        f.extractall(cfg.DATA_DIR)


def fetch_image(row):
    dest = os.path.join(cfg.IMAGE_DIR, row["img_id"] + ".jpg")
    if os.path.exists(dest) and sha256(dest) == row["sha256"]:
        return "have"
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(row["url"], headers=HEADERS), timeout=60).read()
            break
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
    else:
        return "failed: %s" % err
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        return "changed at the source"           # kept out of IMAGE_DIR: the boxes were drawn on the original
    with open(dest, "wb") as f:
        f.write(data)
    time.sleep(0.2)                               # be gentle with the archive server
    return "fetched"


def fetch_images(workers):
    rows = list(csv.DictReader(open(os.path.join(cfg.DATA_DIR, "images.csv"), encoding="utf-8")))
    os.makedirs(cfg.IMAGE_DIR, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(fetch_image, rows))
    counts = {}
    for row, r in zip(rows, res):
        key = r.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if key not in ("have", "fetched"):
            print("  %s: %s (%s)" % (row["img_id"], r, row["url"]))
    print("  images:", ", ".join("%s %d" % kv for kv in sorted(counts.items())))
    return counts.get("have", 0) + counts.get("fetched", 0) == len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", action="store_true", help="region features of the proposer comparison")
    ap.add_argument("--model", action="store_true", help="trained NERGL weights")
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--no_bart", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel image downloads")
    a = ap.parse_args()

    print("dataset (%s) -> %s" % (cfg.HF_DATA_REPO, cfg.DATA_DIR))
    snapshot_download(cfg.HF_DATA_REPO, repo_type="dataset", local_dir=cfg.DATA_DIR, token=TOKEN,
                      allow_patterns=DATA_PARTS)
    for name in FEATURES + (PROPOSER_FEATURES if a.proposers else []):
        fetch_features(name)
    ok = True
    if not a.no_images:
        print("images (ARC) ->", cfg.IMAGE_DIR)
        ok = fetch_images(a.workers)

    print("knowledge cache ->", cfg.KNOWLEDGE_FILE)
    if not os.path.exists(cfg.KNOWLEDGE_FILE):
        os.makedirs(os.path.dirname(cfg.KNOWLEDGE_FILE), exist_ok=True)
        shutil.copyfile(hf_hub_download(cfg.HF_DATA_REPO, KNOWLEDGE, repo_type="dataset", token=TOKEN),
                        cfg.KNOWLEDGE_FILE)

    if a.model:
        dest = os.path.join(cfg.SAVE_DIR, "demo_nergl_vinvl_e50_f0_s16_model_bestgmner")
        print("NERGL weights (%s) -> %s" % (cfg.HF_MODEL_REPO, dest))
        if not os.path.exists(dest):
            os.makedirs(cfg.SAVE_DIR, exist_ok=True)
            shutil.copyfile(hf_hub_download(cfg.HF_MODEL_REPO, MODEL_FILE, token=TOKEN), dest)

    if not a.no_bart:
        print("BART (%s) -> %s" % (BART_REPO, cfg.BART_DIR))
        snapshot_download(BART_REPO, local_dir=cfg.BART_DIR)

    # box labels renamed to the title string; derived from xml/, so built here rather than downloaded
    if os.path.isdir(cfg.XML_DIR) and not os.path.isdir(cfg.XML_LINKED_DIR):
        subprocess.run([sys.executable, os.path.join(ROOT, "data_processing/dataset/link_box_names.py"),
                        cfg.XML_DIR, cfg.XML_LINKED_DIR], check=True)
    print("done" if ok else "done, but some images are missing (listed above)")


if __name__ == "__main__":
    main()
