# coding:utf-8
"""Score every system's NER output against one canonical gold.

Each system previously carried its own copy of the gold inside its prediction
dump, and those copies disagree with the annotation and with each other. This
module reads the gold once from the BIO file and aligns every prediction to it.

Two alignment paths are needed:
  nergl  prediction dumps, whose own gold drops characters outside the BART vocab
  llm    evaluation dumps, which omit images that have no region features
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os
import re
import ast
import json
import argparse
import collections

TUP = re.compile(r"\(([^,]*),\s*\[([^\]]*)\]\s*,\s*<<([^>]*)>>\s*\)")

# Characters the BART tokenizer splits into a bare word-start marker plus a
# second sub-token. train.py reconstructs an entity string from the first
# sub-token of each word only, so for these characters it recovers the marker,
# which decodes to a space and is then stripped. The character is lost from the
# printed string even though the predicted index identifies the word correctly.
# Deleting the same characters from the gold and from every system's output puts
# all systems on one footing; the setting is selected with --norm.
_AFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "affected_chars.json")
AFFECTED = set(json.load(open(_AFF, encoding="utf-8"))) if os.path.exists(_AFF) else set()
NORM = False


def norm(s):
    if not NORM:
        return s
    return "".join(c for c in s if c not in AFFECTED)


# ---------------------------------------------------------------- gold

def read_bio(path):
    recs, cur, toks, labs = [], None, [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("IMGID:"):
            if cur is not None:
                recs.append((cur, toks, labs))
            cur, toks, labs = line[6:], [], []
        elif line.strip():
            p = line.split("\t")
            toks.append(p[0])
            labs.append(p[1])
    if cur is not None:
        recs.append((cur, toks, labs))
    return recs


def bio_entities(toks, labs):
    out, i = [], 0
    while i < len(labs):
        if labs[i].startswith("B-"):
            t = labs[i][2:]
            j = i + 1
            while j < len(labs) and labs[j] == "I-" + t:
                j += 1
            out.append((norm("".join(toks[i:j])), t))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------- predictions

def pred_nergl(path, gold_recs=None):
    """Predicted entities per image, in the order the dump was written.

    Two records in the corpus place the kana reading of a title on its own
    line, which the data pipeline reads as a further instance while a reader
    keyed on IMGID folds it back into the record above. The dump therefore
    holds one block more than the gold for the folds that contain them. Given
    the gold, consecutive blocks are merged until their titles together match
    the gold title, which realigns the two without touching the corpus.
    """
    titles, bags = [], []
    pending_title = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("Pred:"):
            titles.append(pending_title or "")
            bags.append(to_bag([(norm(s.replace(" ", "").strip()), t.strip())
                                for s, _r, t in TUP.findall(line[5:])]))
            pending_title = None
        elif not line.startswith((" ", "\t")) and line.strip():
            pending_title = line
    if gold_recs is None or len(bags) == len(gold_recs):
        return bags

    out, i = [], 0
    for _img, toks, _labs in gold_recs:
        want = "".join("_" if t == " " else t for t in toks)
        acc = collections.Counter() if MULTISET else set()
        seen = ""
        while i < len(bags):
            acc = acc + bags[i] if MULTISET else acc | bags[i]
            seen += titles[i].rstrip("_")
            i += 1
            if seen.replace("_", "") == want.replace("_", "") or len(seen) >= len(want):
                break
        out.append(acc)
    return out


def pred_llm(path):
    """keyed by image id, since these dumps may omit images"""
    out = {}
    img = None
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        if s and not s.startswith((" ", "\t")) and "\t" in s:
            img = s.split("\t")[0].strip()
        elif s.strip().startswith("Pred:"):
            try:
                lst = ast.literal_eval(s.strip()[5:].strip())
            except Exception:
                lst = []
            if img:
                out[img] = to_bag([(norm(t), l) for t, _r, l in lst])
    return out


# ---------------------------------------------------------------- scoring

MULTISET = False


def to_bag(items):
    """a set when mentions are deduplicated, a multiset when they are not"""
    return collections.Counter(items) if MULTISET else set(items)


def overlap(a, b):
    if MULTISET:
        return sum((a & b).values())
    return len(a & b)


def size(a):
    return sum(a.values()) if MULTISET else len(a)


def prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p * 100, r * 100, f * 100


def score_against_gold(gold_recs, preds, keyed):
    """preds: dict img->set when keyed, else ordered list of sets"""
    tp = npred = ngold = 0
    for i, (img, toks, labs) in enumerate(gold_recs):
        g = to_bag(bio_entities(toks, labs))
        if keyed:
            p = preds.get(img, to_bag([]))
        else:
            p = preds[i] if i < len(preds) else to_bag([])
        tp += overlap(g, p)
        npred += size(p)
        ngold += size(g)
    return prf(tp, npred, ngold), ngold, npred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(cfg.TXT_DIR, "test.txt"))
    ap.add_argument("--nergl", nargs="*", default=[])
    ap.add_argument("--llm", nargs="*", default=[])
    ap.add_argument("--multiset", action="store_true",
                    help="count a repeated mention once per occurrence, as "
                         "seqeval and the training metric do (gold 254)")
    ap.add_argument("--norm", action="store_true",
                    help="delete the characters train.py cannot print, in the "
                         "gold and in every prediction alike")
    a = ap.parse_args()

    global NORM, MULTISET
    NORM = a.norm
    MULTISET = a.multiset
    if NORM:
        print("normalising %d characters out of gold and predictions alike\n"
              % len(AFFECTED))

    gold = read_bio(a.gold)
    ngold_total = sum(size(to_bag(bio_entities(t, l))) for _, t, l in gold)
    print(f"canonical gold: {len(gold)} images, {ngold_total} entity mentions\n")
    print(f"{'system':<34}{'P':>8}{'R':>8}{'F1':>8}{'pred':>8}{'gold':>7}")
    print("-" * 73)

    for f in a.nergl:
        (p, r, fs), ng, npd = score_against_gold(gold, pred_nergl(f), False)
        print(f"{os.path.basename(f)[:33]:<34}{p:>8.2f}{r:>8.2f}{fs:>8.2f}{npd:>8}{ng:>7}")
    for f in a.llm:
        (p, r, fs), ng, npd = score_against_gold(gold, pred_llm(f), True)
        print(f"{os.path.basename(f)[:33]:<34}{p:>8.2f}{r:>8.2f}{fs:>8.2f}{npd:>8}{ng:>7}")


if __name__ == "__main__":
    main()
