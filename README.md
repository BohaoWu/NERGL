# NERGL

**NERGL: Named Entity Recognition and Grounding with Large Language Models for Ukiyo-E Artworks**

Source code for our TPDL 2026 paper: [NERGL: Named Entity Recognition and Grounding with Large Language Models for Ukiyo-E Artworks](https://link.springer.com/chapter/10.1007/978-3-032-06136-2_1)

This project performs named entity recognition and visual grounding on Japanese Ukiyo-e (浮世絵) artwork metadata, with RAG-enhanced recognition via LLMs (GPT/Claude) and a Japanese BART model as the backbone.

## Project Structure

```
├── model/                  # Model architecture (BART-based multi-concat)
├── data_processing/        # Data preprocessing, evaluation, and LLM-based RAG
├── train.py / train.sh     # Training scripts
├── test.py / test.sh       # Evaluation scripts
├── demo.py                 # Demo script
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
pip install torch transformers fastNLP fitlog
```

3. Configure API keys (optional, for RAG mode):

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

4. Prepare data and models:

- Place the Ukiyo-e dataset under `Ukiyoe1000/` (txt/ and xml/ subdirectories)
- Place VinVL visual features under `Ukiyoe1000_VinVL/`
- Download a Japanese BART model to `download_model/`

## Usage

### Training

```bash
bash train.sh
```

### Evaluation

```bash
bash test.sh
```

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

- [GMNER](https://github.com/NUSTM/GMNER) (ACL 2023: [Grounded Multimodal Named Entity Recognition on Social Media](https://aclanthology.org/2023.acl-long.508.pdf))
- [BARTNER](https://github.com/yhcc/BARTNER)
