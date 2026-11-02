# coding:utf-8
"""Regenerate the LLM NER knowledge cache consumed by rag_method=1.

Output format matches the existing cache: "<img_id>###<ner result string>" per line.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing.gpt_request import GPTClient


def read_titles(txt_dir):
    items = []
    for split in ("train", "dev", "test"):
        path = os.path.join(txt_dir, split + ".txt")
        cur, toks = None, []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("IMGID:"):
                    if cur is not None:
                        items.append((cur, "".join(toks)))
                    cur, toks = line.replace("IMGID:", ""), []
                elif line.strip():
                    toks.append(line.split("\t")[0])
        if cur is not None:
            items.append((cur, "".join(toks)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", default=cfg.TXT_DIR + "/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    items = read_titles(args.txt_dir)
    print(f"{len(items)} titles; model={args.model}")

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if "###" in line:
                k, _ = line.split("###", 1)
                done[k.strip()] = True
        print(f"resuming, {len(done)} already cached")

    todo = [(i, t) for i, t in items if i not in done]
    print(f"{len(todo)} to generate")

    client = GPTClient(gpt_model=args.model)
    lock_out = open(args.out, "a", encoding="utf-8")

    def work(item):
        img_id, title = item
        try:
            res = client.getNERResultStrResponseFomGPT(title)
            return img_id, (res or "").replace("\n", " ").strip()
        except Exception as e:
            print(f"  error {img_id}: {type(e).__name__}")
            return img_id, ""

    n = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            img_id, res = fut.result()
            lock_out.write(f"{img_id}###{res}\n")
            lock_out.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    lock_out.close()
    # Kept beside the cache, whose lines data_pipe parses as img_id###result.
    with open(args.out + ".snapshots.json", "w") as f:
        json.dump(dict(client.snapshots), f, indent=2)
    print(f"Done -> {args.out}; snapshots {dict(client.snapshots)}")


if __name__ == "__main__":
    main()
