# coding:utf-8
"""Five-fold splits over all 1086 prints.

The published split tests on 108 prints, which is too few to separate the
configurations: the seed to seed spread was as large as the spread between
configurations. Under five folds every print is tested exactly once, so the
test evidence is the whole corpus instead of a tenth of it. Each fold holds out
one fifth for test and takes a tenth of the remainder for development, which
keeps the training set close to its original size.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os, random, argparse



def read_blocks(path):
    out, cur = [], []
    for line in open(path, encoding='utf-8'):
        if line.startswith('IMGID:'):
            if cur:
                out.append(cur)
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--txt_dir', default=cfg.TXT_DIR)
    ap.add_argument('--out_dir', default=cfg.CV_DIR)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    blocks = []
    for s in ('train', 'dev', 'test'):
        blocks += read_blocks(os.path.join(a.txt_dir, '%s.txt' % s))
    blocks = [b for b in blocks if b[0].strip() != 'IMGID:']
    print('prints pooled: %d' % len(blocks))

    random.Random(a.seed).shuffle(blocks)
    folds = [blocks[i::a.folds] for i in range(a.folds)]

    for k in range(a.folds):
        test = folds[k]
        rest = [b for j in range(a.folds) if j != k for b in folds[j]]
        ndev = max(1, len(rest) // 10)
        # Shuffle the remainder once per fold before cutting the development
        # set off it. Taking the head of the remainder instead gives folds 1
        # onwards the same development prints, because the remainder starts at
        # fold 0 for every one of them, which couples their model selection.
        random.Random(a.seed + 1000 + k).shuffle(rest)
        dev, train = rest[:ndev], rest[ndev:]
        d = os.path.join(a.out_dir, 'fold%d' % k)
        os.makedirs(d, exist_ok=True)
        for name, part in (('train', train), ('dev', dev), ('test', test)):
            with open(os.path.join(d, '%s.txt' % name), 'w', encoding='utf-8') as f:
                for b in part:
                    f.writelines(b)
                    if not b[-1].endswith('\n'):
                        f.write('\n')
        print('fold %d: train %4d  dev %3d  test %3d' % (k, len(train), len(dev), len(test)))


if __name__ == '__main__':
    main()
