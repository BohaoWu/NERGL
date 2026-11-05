NERGL: Named Entity Recognition and Grounding with Large Language Models for Ukiyo-E Artworks
=============================================================================================

📄 [Paper](https://link.springer.com/chapter/10.1007/978-3-032-06136-2_1)  |  🤗 [Model](https://huggingface.co/BohaoWu/NERGL)  |  🗂️ [Dataset](https://huggingface.co/datasets/BohaoWu/Ukiyoe_MNERG)  |  📜 [Demo](https://huggingface.co/spaces/BohaoWu/NERGL-demo)

Official code for our TPDL 2025 paper and its journal extension.

NERGL finds the named entities in the title of an ukiyo-e (浮世絵) print, types them, and links each one
to the region of the print that depicts it, or marks it as not depicted. A Japanese BART model reads
the title, the answer of a large language model (LLM) to the same title, and candidate regions from a
region proposer.

## Dataset

The Ukiyo-e MNERG dataset is on Hugging Face:
[BohaoWu/Ukiyoe_MNERG](https://huggingface.co/datasets/BohaoWu/Ukiyoe_MNERG).

- 1,086 prints (736 yakusha-e, 350 meisho-e) from the
  [Japanese Prints (Ukiyo-e) and Paintings Portal Database](https://www.dh-jac.net/db/nishikie/search_portal.php?lang=en)
  of the Art Research Center, Ritsumeikan University
- 2,549 entity mentions of four types: Character (替名), Act (演目), Actor (役者), and Location (地名)
- box annotations of the depicted entities, and the five-fold split used in the article

| Path | Content |
|---|---|
| `txt/` | the original train / dev / test split, BIO tags per character |
| `cv/fold{0..4}/` | the five-fold split: about 783 train, 86 dev, and 217 test prints per fold |
| `xml/` | box annotations, one Pascal VOC file per print; the box label is the entity string |
| `features/Ukiyoe1000_*.zip` | region proposals and features of VinVL and the other proposers in the comparison |
| `knowledge/gpt_ner_res_sol_v2.txt` | the LLM output: the answer of GPT-6.1 Sol for each title |
| `images.csv` | source URL and SHA-256 of each print's image |

The images are not redistributed. `script/download.py` fetches them from the Art Research Center and
checks each hash (see [Setup](#setup)). Please use them under the Center's terms of use. The trained
weights of one run (fold 0, seed 16) are at [BohaoWu/NERGL](https://huggingface.co/BohaoWu/NERGL).

## Project Structure

```
├── model/                  # Model architecture (BART-based multi-concat)
├── data_processing/        # Data preprocessing, evaluation, and LLM-based RAG
├── script/                 # train.py, test.py, and run.sh (every experiment in the article)
└── .env.example            # API key configuration template
```

## Requirements

- Python 3.8+
- PyTorch 2.x (CUDA)
- transformers >= 4.x
- fastNLP 0.6.0
- fitlog

## Setup

1. Clone the repository:

```bash
git clone https://github.com/BohaoWu/NERGL.git
cd NERGL
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API keys (optional, for RAG mode):

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

4. Download the data and models:

```bash
python3 script/download.py          # dataset, CV splits, VinVL features, knowledge cache, BART
```

It fetches the annotations, splits, region features and knowledge cache from the dataset
[BohaoWu/Ukiyoe_MNERG](https://huggingface.co/datasets/BohaoWu/Ukiyoe_MNERG), and BART from
[ku-nlp/bart-large-japanese](https://huggingface.co/ku-nlp/bart-large-japanese). The prints are not
redistributed: they are fetched from the Art Research Center, Ritsumeikan University, and checked
against the SHA-256 recorded with the annotations. `--proposers` adds the region features of the
proposer comparison and `--model` the trained weights ([BohaoWu/NERGL](https://huggingface.co/BohaoWu/NERGL)).

Every path (dataset, models, outputs, logs) is set in `data_processing/config.py` and can be changed
in `.env` (see `.env.example`). Relative paths are read from the repository root, so the clone can
live anywhere. `data/`, `download_model/` and `saved_model/` are not in the repository.

## Usage

### Training

```bash
bash script/run.sh nergl          # every experiment: see script/README.md
```

A single run calls `python3 script/train.py ...` from the repository root; `script/run.sh` shows the full arguments.

### Evaluation

```bash
python3 data_processing/evaluation/score_cv.py --table all
```

## Results

Five-fold cross-validation over all 1086 prints (2549 entity mentions), three seeds per fold; the
prediction counts of the fifteen runs are pooled before scoring. These are the numbers of the journal
version of the article. `data_processing/evaluation/score_cv.py` reproduces the trained systems and
`data_processing/evaluation/score_llm.py` the LLMs.

### NER and GMNER

| System | NER P | NER R | NER F1 | GMNER P | GMNER R | GMNER F1 |
|--------|------:|------:|-------:|--------:|--------:|---------:|
| BERT | 79.06 | 80.62 | 79.83 | 32.39\* | 33.03\* | 32.71\* |
| BERT-CRF | 79.13 | 80.20 | 79.66 | 32.49\* | 32.93\* | 32.71\* |
| MAF | 80.54 | 82.14 | 81.33 | 33.12\* | 33.78\* | 33.45\* |
| HVPNeT | 82.23 | 82.23 | 82.23 | 33.71\* | 33.71\* | 33.71\* |
| ITA | 81.02 | 82.73 | 81.86 | 33.54\* | 34.25\* | 33.89\* |
| MKGformer | 81.29 | 81.55 | 81.42 | 33.27\* | 33.37\* | 33.32\* |
| H-Index (NERGL without the LLM output) | 76.74 | 77.63 | 77.18 | 57.75 | 58.42 | 58.08 |
| **NERGL** | **82.32** | **83.07** | **82.69** | **62.14** | **62.70** | **62.42** |
| GPT-6.1 Sol | 76.68 | 77.68 | 77.18 | 59.33 | 60.10 | 59.72 |
| GPT-5.5 | 74.98 | 77.13 | 76.04 | 40.96 | 42.13 | 41.54 |
| GPT-6 Luna | 72.65 | 73.68 | 73.16 | 34.70 | 35.19 | 34.94 |
| GPT-5.4 | 75.76 | 73.95 | 74.85 | 29.14 | 28.44 | 28.79 |
| GPT-4.1 | 56.05 | 66.10 | 60.67 | 26.28 | 30.99 | 28.44 |

\* NER-only systems, scored by answering "not depicted" for every recognised entity. The LLMs receive the
image, the title, and the same 18 VinVL candidate regions as NERGL, with no training on our data.

### Region proposers

The full model retrained for each configuration (30 epochs, so slightly below the table above).
Coverage is the share of the 1460 annotated entities that some candidate region covers at IoU ≥ 0.5.

| Proposer | Regions per image | Coverage (%) | NER F1 | GMNER F1 |
|----------|------------------:|-------------:|-------:|---------:|
| GDINO-P (Grounding DINO, 30 object types) | 17.9 | 91.0 | 82.28 | **61.56** |
| Grounding DINO (10 object types) | 17.8 | **92.1** | **82.47** | 61.29 |
| VinVL | 18.4 | 90.9 | 82.02 | 60.81 |
| Florence-2 | 7.2 | 90.6 | 82.04 | 60.14 |
| RT-DETR-x | 17.8 | 87.8 | 82.03 | 56.95 |
| CLIP ViT-L/14 (on the YOLOv8x boxes) | 3.4 | 59.5 | 81.33 | 52.63 |
| YOLOv8x | 3.4 | 59.5 | 81.11 | 51.76 |

Coverage, not the number of regions, largely determines the grounding result (Spearman's ρ = 0.95
between coverage and GMNER F1).

## RAG Configuration

RAG mode can be configured in `data_processing/config.py`:

- `switch_rag`: Enable/disable RAG enhancement
- `rag_method`: RAG source (0: disabled, 1: from dict file, 2: GPT directly, 5: Claude)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-032-06136-2_1,
  author="Wu, Bohao and Maeda, Akira",
  title="NERGL: Named Entity Recognition and Grounding with Large Language Models for Ukiyo-E Artworks",
  booktitle="New Trends in Theory and Practice of Digital Libraries",
  year="2026",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="3--13",
  isbn="978-3-032-06136-2"
}
```

## Acknowledgements

- [GMNER](https://github.com/NUSTM/GMNER) (ACL 2023: [Grounded Multimodal Named Entity Recognition on Social Media](https://aclanthology.org/2023.acl-long.508.pdf)), on which this code is based
- [BARTNER](https://github.com/yhcc/BARTNER)
- The prints and titles come from the [Japanese Prints (Ukiyo-e) and Paintings Portal Database](https://www.dh-jac.net/db/nishikie/search_portal.php?lang=en) of the Art Research Center, Ritsumeikan University
- [ku-nlp/bart-large-japanese](https://huggingface.co/ku-nlp/bart-large-japanese) (backbone) and [VinVL](https://github.com/pzzhang/VinVL) (default region proposer)
- Baselines: [MAF](https://github.com/xubodhu/MAF), [HVPNeT](https://github.com/zjunlp/HVPNeT), and [MKGformer](https://github.com/zjunlp/MKGformer) (official code), and ITA (our reimplementation, following [Alibaba-NLP/KB-NER](https://github.com/Alibaba-NLP/KB-NER))

This work was supported by JST SPRING, Grant Number JPMJSP2101, and JSPS KAKENHI Grant Number 23K11780.
