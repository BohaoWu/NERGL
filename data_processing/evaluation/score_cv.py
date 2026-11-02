# coding:utf-8
"""Pool every run of the five-fold, three-seed protocol into one score.

Counts add across folds and seeds, so precision and recall are computed once on
the pooled predictions rather than averaged over runs. Names follow script/run.sh:

  NERGL and its variants  saved_model/pred_<prefix>_<proposer>_e<E>_f<K>_s<S>_model_best{ner,gmner}.txt
  BERT, BERT-CRF          saved_model/<bert|bertcrf>_f<K>_s<S>.json

NER is scored on the dump written at the epoch with the best validation NER F1,
and GMNER on the dump written at the epoch with the best validation GMNER F1.
A baseline that makes no region decision is given the answer "not depicted" for
every entity it recognises.

Usage: python3 data_processing/evaluation/score_cv.py [--seeds "16 42 2023"] [--table 2|3|4|5|all]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os, re, sys, ast, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_ner_unified as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CVDIR = cfg.CV_DIR
TYPES = ['替名', '演目', '役者', '地名']
TYPE_EN = {'替名': 'Character', '演目': 'Act', '役者': 'Actor', '地名': 'Location'}
TUP = re.compile(r"\(([^,]*),\s*\[([^\]]*)\]\s*,\s*<<([^>]*)>>\s*\)")
NO_REGION = 18          # region index the metric uses for "not depicted"


def prf(tp, npred, ngold):
    return S.prf(tp, npred, ngold)


# ------------------------------------------------------------- gold
_GOLD = {}


def gold(k):
    if k not in _GOLD:
        _GOLD[k] = S.read_bio(os.path.join(CVDIR, 'fold%d' % k, 'test.txt'))
    return _GOLD[k]


def gold_bags(k):
    return [collections.Counter(S.bio_entities(t, l)) for _, t, l in gold(k)]


# ------------------------------------------------------------- NERGL dumps
def read_dump(path):
    """blocks of (title, pred tuples, gold tuples); a tuple is (text, regions, type)"""
    blocks, title, pred = [], '', None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('Pred:'):
            pred = TUP.findall(line[5:])
        elif re.match(r'\s*GT\s*:', line):
            gt = TUP.findall(line.split(':', 1)[1])
            blocks.append((title, pred or [], gt))
            pred = None
        elif line.strip() and not line.startswith((' ', '\t', '{')):
            title = line
    return blocks


def tuples(raw):
    out = []
    for s, r, t in raw:
        regs = frozenset(int(x) for x in r.split(',') if x.strip().lstrip('-').isdigit())
        out.append((S.norm(s.replace(' ', '').strip()), regs, t.strip()))
    return out


def align(blocks, k):
    """merge the extra blocks that the kana reading lines create (see S.pred_nergl)"""
    G = gold(k)
    if len(blocks) == len(G):
        return [(tuples(p), tuples(g)) for _t, p, g in blocks]
    out, i = [], 0
    for _img, toks, _labs in G:
        want = ''.join('_' if t == ' ' else t for t in toks).replace('_', '')
        P, Gt, seen = [], [], ''
        while i < len(blocks):
            t, p, g = blocks[i]
            P += tuples(p); Gt += tuples(g); seen += t.rstrip('_'); i += 1
            if seen.replace('_', '') == want or len(seen) >= len(want):
                break
        out.append((P, Gt))
    return out


def match(P, G, key):
    """greedy one-to-one matching of predicted and gold tuples on key()"""
    left = list(G)
    n = 0
    for p in P:
        for j, g in enumerate(left):
            if key(p, g):
                n += 1
                del left[j]
                break
    return n


def nergl_counts(path, k):
    """per-print counts from one dump: entity, region, joint, and GMNER"""
    c = collections.Counter()
    for P, G in align(read_dump(path), k):
        c['npred'] += len(P); c['ngold'] += len(G)
        c['ent'] += match(P, G, lambda p, g: p[0] == g[0] and p[2] == g[2])
        c['reg'] += match(P, G, lambda p, g: p[0] == g[0] and bool(p[1] & g[1]))
    # The joint count comes from the metric line at the end of the dump, which
    # matches spans by position; matching by text alone credits a few spans that
    # repeat a string elsewhere in the title.
    last = open(path, encoding='utf-8').read().strip().splitlines()[-1]
    d = ast.literal_eval(last)
    c['both'] = d['useful_correct'] + d['noregion_correct']
    return c


def nergl_ner_bags(path, k):
    return [collections.Counter((p[0], p[2]) for p in P) for P, _G in align(read_dump(path), k)]


def gold_noregion_bags(k, ref_dump):
    """gold mentions with no annotated region, read from any NERGL dump of the fold"""
    return [collections.Counter((g[0], g[2]) for g in G if g[1] == {NO_REGION})
            for _P, G in align(read_dump(ref_dump), k)]


# ------------------------------------------------------------- baselines
def text_bags(tag, k, s):
    p = '%s/%s_f%d_s%s.json' % (cfg.SAVE_DIR, tag, k, s)
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding='utf-8'))['preds']
    out = []
    for img, toks, _ in gold(k):
        labs = d.get(img, [])
        n = min(len(toks), len(labs))
        out.append(collections.Counter(S.bio_entities(toks[:n], labs[:n])))
    return out


# ------------------------------------------------------------- pooling
class Pool:
    def __init__(self):
        self.c = collections.Counter()
        self.t = collections.defaultdict(collections.Counter)
        self.runs = 0

    def add_ner(self, pred_bags, k, noreg=None):
        for p, g in zip(pred_bags, gold_bags(k)):
            self.c['tp'] += sum((p & g).values())
            self.c['npred'] += sum(p.values()); self.c['ngold'] += sum(g.values())
            for ty in TYPES:
                pt = collections.Counter({e: n for e, n in p.items() if e[1] == ty})
                gt = collections.Counter({e: n for e, n in g.items() if e[1] == ty})
                self.t[ty]['tp'] += sum((pt & gt).values())
                self.t[ty]['npred'] += sum(pt.values()); self.t[ty]['ngold'] += sum(gt.values())
        if noreg is not None:
            for p, nr in zip(pred_bags, noreg):
                self.c['null_tp'] += sum((p & nr).values())

    def ner(self):
        return prf(self.c['tp'], self.c['npred'], self.c['ngold'])

    def null_gmner(self):
        return prf(self.c['null_tp'], self.c['npred'], self.c['ngold'])

    def per_type(self):
        f = {ty: prf(self.t[ty]['tp'], self.t[ty]['npred'], self.t[ty]['ngold'])[2] for ty in TYPES}
        n = sum(self.t[ty]['ngold'] for ty in TYPES)
        w = sum(f[ty] * self.t[ty]['ngold'] for ty in TYPES) / n if n else 0.0
        return f, w


def dump_path(prefix, prop, epochs, k, s, sel):
    return '%s/pred_%s_%s_e%d_f%d_s%s_model_best%s.txt' % (cfg.SAVE_DIR, prefix, prop, epochs, k, s, sel)


def score_nergl(prefix, prop, epochs, folds, seeds):
    ner, gm, missing = Pool(), collections.Counter(), 0
    for k in folds:
        for s in seeds:
            pn, pg = (dump_path(prefix, prop, epochs, k, s, x) for x in ('ner', 'gmner'))
            if not (os.path.exists(pn) and os.path.exists(pg)):
                missing += 1
                continue
            ner.add_ner(nergl_ner_bags(pn, k), k)
            gm.update(nergl_counts(pg, k))
            ner.runs += 1
    return ner, gm, missing


def score_baseline(loader, folds, seeds, ref):
    pool, missing = Pool(), 0
    for k in folds:
        noreg = gold_noregion_bags(k, ref(k)) if ref(k) else None
        for s in seeds:
            b = loader(k, s)
            if b is None:
                missing += 1
                continue
            pool.add_ner(b, k, noreg)
            pool.runs += 1
    return pool, missing


# ------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0 1 2 3 4')
    ap.add_argument('--seeds', default='16 42 2023')
    ap.add_argument('--table', default='all')
    a = ap.parse_args()
    S.NORM, S.MULTISET = True, True
    folds = [int(x) for x in a.folds.split()]
    seeds = a.seeds.split()
    T = a.table

    def ref(k):   # any finished NERGL dump of the fold supplies the gold regions
        for s in seeds:
            p = dump_path('nergl', 'vinvl', 50, k, s, 'gmner')
            if os.path.exists(p):
                return p
        return None

    rows = {}
    for lbl, prefix in (('NERGL', 'nergl'), ('NERGL without knowledge stream', 'nollm')):
        rows[lbl] = score_nergl(prefix, 'vinvl', 50, folds, seeds)
    base = {'BERT': lambda k, s: text_bags('bert', k, s),
            'BERT-CRF': lambda k, s: text_bags('bertcrf', k, s)}
    for tag, lbl in (('maf', 'MAF'), ('hvpnet', 'HVPNeT'), ('ita', 'ITA'), ('mkgformer', 'MKGformer')):
        base[lbl] = (lambda t: lambda k, s: text_bags(t, k, s))(tag)
    brows = {lbl: score_baseline(f, folds, seeds, ref) for lbl, f in base.items()}

    if T in ('2', 'all'):
        print('Table 2  NER and GMNER, pooled over %d folds x %d seeds' % (len(folds), len(seeds)))
        print('%-32s %6s %6s %6s   %6s %6s %6s  runs' % ('System', 'P', 'R', 'F1', 'P', 'R', 'F1'))
        for lbl, (p, m) in brows.items():
            if not p.runs:
                continue
            n, g = p.ner(), p.null_gmner()
            print('%-32s %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f  %d%s' % ((lbl,) + n + g + (p.runs, ' (null)')))
        for lbl, (ner, gm, miss) in rows.items():
            n = ner.ner()
            g = prf(gm['both'], gm['npred'], gm['ngold'])
            print('%-32s %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f  %d' % ((lbl,) + n + g + (ner.runs,)))
        print()

    if T in ('3', 'all'):
        print('Table 3  F1 by entity type (NER-selected epoch)')
        print('%-32s %9s %9s %9s %9s %9s' % ('System', 'Character', 'Act', 'Actor', 'Location', 'Weighted'))
        allrows = [(l, p) for l, (p, _m) in brows.items() if p.runs] + [(l, r[0]) for l, r in rows.items()]
        for lbl, p in allrows:
            f, w = p.per_type()
            print('%-32s %9.2f %9.2f %9.2f %9.2f %9.2f' % ((lbl,) + tuple(f[t] for t in TYPES) + (w,)))
        print()

    if T in ('4', 'all'):
        print('Table 4  detection rates over gold mentions (GMNER-selected epoch)')
        for lbl, (_ner, gm, _m) in rows.items():
            if gm['ngold']:
                print('%-32s entity %5.1f  region %5.1f  both %5.1f' % (
                    lbl, 100 * gm['ent'] / gm['ngold'], 100 * gm['reg'] / gm['ngold'],
                    100 * gm['both'] / gm['ngold']))
        print()

    if T in ('5', 'all'):
        print('Table 5  region proposers, 30 epochs')
        print('%-12s %6s %6s  runs' % ('Proposer', 'NER', 'GMNER'))
        for prop in ('vinvl', 'florence', 'gdinop', 'gdino', 'rtdetr', 'clip', 'yolo'):
            ner, gm, miss = score_nergl('nergl', prop, 30, folds, seeds)
            if not ner.runs:
                print('%-12s  (not run yet)' % prop); continue
            print('%-12s %6.2f %6.2f  %d%s' % (prop, ner.ner()[2], prf(gm['both'], gm['npred'], gm['ngold'])[2],
                                            ner.runs, '' if not miss else ' (%d missing)' % miss))


if __name__ == '__main__':
    main()
