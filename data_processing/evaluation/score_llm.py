# coding:utf-8
"""Score LLM GMNER dumps over the whole corpus, on the same footing as score_cv.py.

NER uses the unified scorer (canonical gold from the five test folds, rare
characters removed, mentions counted as a multiset). GMNER and the detection
rates read the gold regions that each dump carries; a prediction is correct when
its string and type match a gold mention and its region index is one of that
mention's regions (the no-region marker counts as a region). Prints without an
image are absent from the dumps, so their mentions count as missed.

Usage: python3 data_processing/evaluation/score_llm.py saved_model/llm_v2/<model>_gmner_result.txt ...
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os, sys, ast, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_ner_unified as S

ALL = os.path.join(cfg.CV_DIR, 'all.txt')


def blocks(path):
    out, img = {}, None
    for line in open(path, encoding='utf-8'):
        s = line.rstrip('\n')
        if s and not s.startswith((' ', '\t')) and '\t' in s:
            img = s.split('\t')[0].strip()
        elif s.strip().startswith(('Pred:', 'GT:')) and img:
            key, lst = s.strip().split(':', 1)
            try:
                v = ast.literal_eval(lst.strip())
            except Exception:
                v = []
            out.setdefault(img, {})[key] = [(S.norm(t), frozenset(r), l) for t, r, l in v]
    return out


def match(P, G, key):
    left, n = list(G), 0
    for p in P:
        for j, g in enumerate(left):
            if key(p, g):
                n += 1; del left[j]; break
    return n


def main():
    S.NORM, S.MULTISET = True, True
    gold = S.read_bio(ALL)
    ngold = sum(len(S.bio_entities(t, l)) for _, t, l in gold)
    print('%-14s %6s %6s %6s   %6s %6s %6s   %6s %6s %6s' % (
        'model', 'NER P', 'R', 'F1', 'GM P', 'R', 'F1', 'ent%', 'reg%', 'both%'))
    for path in sys.argv[1:]:
        name = os.path.basename(path).replace('_gmner_result.txt', '')
        ner = S.pred_llm(path)
        tp = npred = 0
        for img, toks, labs in gold:
            g = collections.Counter(S.bio_entities(toks, labs))
            p = ner.get(img, collections.Counter())
            tp += sum((g & p).values()); npred += sum(p.values())
        c = collections.Counter()
        for img, b in blocks(path).items():
            P, G = b.get('Pred', []), b.get('GT', [])
            c['npred'] += len(P)
            c['ent'] += match(P, G, lambda p, g: p[0] == g[0] and p[2] == g[2])
            c['reg'] += match(P, G, lambda p, g: p[0] == g[0] and bool(p[1] & g[1]))
            c['both'] += match(P, G, lambda p, g: p[0] == g[0] and p[2] == g[2] and bool(p[1] & g[1]))
        n = S.prf(tp, npred, ngold)
        m = S.prf(c['both'], c['npred'], ngold)
        print('%-14s %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f   %6.1f %6.1f %6.1f' % (
            (name,) + n + m + (100 * c['ent'] / ngold, 100 * c['reg'] / ngold, 100 * c['both'] / ngold)))
    print('gold mentions: %d' % ngold)


if __name__ == '__main__':
    main()
