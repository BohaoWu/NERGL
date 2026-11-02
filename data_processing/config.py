"""Every path and switch of the project, in one place.

A value is taken from the environment first, then from .env in the repository
root, then from the default below. A relative path is read from the repository
root, so the clone can live anywhere. `python3 data_processing/config.py --shell`
prints the paths for script/run.sh.
"""
import os
import sys

# the repository root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env(path=os.path.join(ROOT, ".env")):
    """KEY=value lines of .env, without overriding the environment"""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.split(" #")[0].strip().strip("\"'"))


_load_env()


def _path(name, default):
    p = os.environ.get(name, default)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


### data: the dataset (script/download.sh puts it here)
DATA_DIR = _path("DATA_DIR", "data/Ukiyoe1000")
TXT_DIR = os.path.join(DATA_DIR, "txt")                 # published train/dev/test split
XML_DIR = os.path.join(DATA_DIR, "xml")                 # box annotations
XML_LINKED_DIR = os.path.join(DATA_DIR, "xml_linked")   # box labels renamed to the title string
CV_DIR = os.path.join(DATA_DIR, "cv")                   # five-fold splits
IMAGE_DIR = os.path.join(DATA_DIR, "ukiyoe_picture")
VINVL_DIR = os.path.join(DATA_DIR, "Ukiyoe1000_VinVL")  # region proposals; the others sit beside it

# where script/download.py fetches from: the dataset, and the trained weights
HF_DATA_REPO = os.environ.get("HF_DATA_REPO", "BohaoWu/Ukiyoe_MNERG")    # a dataset repository
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "BohaoWu/NERGL")

### models and outputs
MODEL_DIR = _path("MODEL_DIR", "download_model")
BART_DIR = os.path.join(MODEL_DIR, "bart-large-japanese")
DETECTOR_DIR = os.path.join(MODEL_DIR, "detectors")
SAVE_DIR = _path("SAVE_DIR", "saved_model")
LOG_DIR = _path("LOG_DIR", "log")
FIGURE_DIR = _path("FIGURE_DIR", "figure")
# ports of the third-party baselines; not in the repository, and script/run.sh
# skips a baseline whose code is not found here
BASELINE_DIR = _path("BASELINE_DIR", "baselines")

### LLM knowledge stream
# the cache read by rag_method=1; RAG_GPT_FILE lets parallel runs point at different caches
KNOWLEDGE_FILE = _path("KNOWLEDGE_FILE", "saved_model/gpt_ner_res_sol_v2.txt")
rag_gpt_file = _path("RAG_GPT_FILE", KNOWLEDGE_FILE)
rag_gpt_dict_file = rag_gpt_file
raw_file = os.path.join(SAVE_DIR, "raw_file.txt")
switch_rag = True
# how to get rag infomation
# 0:Without RAG
# 1:from dict file
# 2:from gpt dirrectly
# 5:Get description from named entity by claude
rag_method = int(os.environ.get("RAG_METHOD", "1"))

# data_pipe.py
cur_iou_score_thresh = 0.5

### original metadata of ukiyoe (Label Studio export, dataset/ scripts)
META_DIR = _path("META_DIR", "data/demo_data")
original_data = os.path.join(META_DIR, "original_data.json")
outputfile_path = os.path.join(META_DIR, "tmp.json")
download_url = os.path.join(META_DIR, "download_url.txt")
picture_path = IMAGE_DIR + "/"
figure_path = FIGURE_DIR + "/"

SHELL_VARS = ["ROOT", "DATA_DIR", "CV_DIR", "XML_DIR", "XML_LINKED_DIR", "IMAGE_DIR", "VINVL_DIR",
              "MODEL_DIR", "BART_DIR", "SAVE_DIR", "LOG_DIR", "KNOWLEDGE_FILE", "BASELINE_DIR"]

if __name__ == "__main__" and "--shell" in sys.argv:
    for k in SHELL_VARS:
        print("%s='%s'" % (k, globals()[k]))
