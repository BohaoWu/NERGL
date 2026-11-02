# Experiments

`run.sh` runs every experiment in the article. The first argument picks the
system; options pick the folds, seeds, GPUs, epochs and region proposers. It
expects the five-fold splits that `data_processing/dataset/make_cv_folds.py` writes to
`data/Ukiyoe1000/cv/fold{0..4}/` and the knowledge caches in `saved_model/`. A job
whose output is already complete is skipped, so an interrupted run can be
restarted with the same command.

| Command | Produces | Runs | Time |
|---|---|---|---|
| `script/run.sh nergl` | Tables 2-4, NERGL | 5 folds x 3 seeds, 50 epochs | ~3 h on 2 GPUs |
| `script/run.sh nollm` | Table 2, GMNER baseline (empty knowledge stream, as in H-Index) | 5 folds x 3 seeds, 50 epochs | ~3 h on 2 GPUs |
| `script/run.sh bert`, `bertcrf`, `maf`, `hvpnet`, `ita`, `mkgformer` | Tables 2-3, baselines | 5 folds x 3 seeds each | ~1-1.5 h together |
| `script/run.sh proposers` | Table 5 | 7 proposers x 5 folds x 3 seeds, 30 epochs | ~10-12 h on 2 GPUs |
| `script/run.sh llm` | Tables 2 and 4, LLMs | 5 models over 1086 prints; o3 keeps its earlier run | ~1 h, paid API |
| `script/run.sh all` | every task above in turn | | |

Options and their defaults:

```
--folds "0 1 2 3 4"   --seeds "16 42 2023"   --gpus "1 2"
--epochs N            50 for nergl, 30 for proposers
--proposers LIST      vinvl for nergl, all for proposers
                      (vinvl gdino gdinop florence rtdetr yolo clip)
--models LIST         five LLMs (o3 is not rerun)
--ckpt last|none      keep the final checkpoint of each run, or none
--dry                 print the job list and exit
```

For example, `script/run.sh proposers --proposers "gdino gdinop" --gpus "1 2"`
reruns only the two Grounding DINO configurations.

Notes:

- The baselines run third-party code that is not in this repository. `run.sh`
  looks for it in `BASELINE_DIR` (`data_processing/config.py`, default
  `baselines/`): `text/run_text_baselines.py` for BERT and BERT-CRF, and
  `<method>/run_fold.sh` for the MNER baselines. A baseline whose code is not
  there is skipped with a message, as is a proposer whose features have not
  been downloaded (`script/download.py --proposers`) and the LLM task when
  `OPENAI_API_KEY` is not set. A fresh clone therefore runs
  `script/run.sh all` through, and `score_cv.py` leaves out the systems that
  have no runs.
- Each NERGL run writes two prediction files, `pred_<run>_bestner.txt` and
  `pred_<run>_bestgmner.txt`, taken at the epoch with the best validation NER F1
  and the best validation GMNER F1. Tables 3 and 4 are computed from them.
- Grounding DINO uses its own decoder query vectors. If
  `Ukiyoe1000_GDino_VL/` or `Ukiyoe1000_GDino_Prompt_VL/` is incomplete, the
  script extracts them first on GPU1, the RTX A6000 that produced the boxes;
  another GPU model reorders near-tied boxes.
- `llm` reads the API key from `.env`, builds `data/Ukiyoe1000/cv/all.txt` from the
  five test folds, and records the snapshot that served each request at the end
  of every result file.

Scoring is separate from running. `data_processing/evaluation/score_cv.py` pools the folds
from the training logs, and `data_processing/evaluation/score_ner_unified.py` scores any
prediction dump against one canonical gold.
