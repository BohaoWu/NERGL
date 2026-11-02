#!/usr/bin/env bash
# One driver for every experiment in the article. The task picks the system, and
# the options pick folds, seeds, GPUs, epochs and region proposers.
#
#   script/run.sh nergl     [--proposers vinvl] [--epochs 50]     Tables 2-4, NERGL
#   script/run.sh proposers [--proposers all]   [--epochs 30]     Table 5
#   script/run.sh nollm     [--epochs 50]     Table 2, GMNER baseline: NERGL with an
#                                          empty knowledge stream, as in H-Index
#   script/run.sh bert | bertcrf                                  Tables 2-3, baselines
#   script/run.sh maf | hvpnet | ita | mkgformer                  Tables 2-3, MNER baselines
#   script/run.sh llm       [--models "gpt-6.1-sol ..."]          Tables 2 and 4, LLMs
#   script/run.sh all                                             every task above in turn
#
# The baselines run third-party code that is not in the repository. Its ports are
# looked up in BASELINE_DIR (data_processing/config.py, default baselines/), and a
# baseline whose code is not there is skipped, so a fresh clone runs the rest.
#
# Common options (defaults in brackets):
#   --folds "0 1 2 3 4"   --seeds "16 42 2023"   --gpus "1 2"
#   --ckpt last|none      keep the final checkpoint of each run, or none [last]
#   --dry                 print the jobs and exit
#
# Jobs are dealt round-robin to the GPUs, one at a time per GPU. A job whose
# output is already complete is skipped, so an interrupted run can be restarted.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)     # the repository root
# Paths come from data_processing/config.py (environment > .env > defaults).
eval "$(python3 "$ROOT/data_processing/config.py" --shell)"
U=$DATA_DIR; BART=$BART_DIR; SOL=$KNOWLEDGE_FILE
cd $ROOT
mkdir -p $LOG_DIR/train $LOG_DIR/eval $LOG_DIR/run $LOG_DIR/fitlog $LOG_DIR/run/done $SAVE_DIR
# Box labels that name a title entity in another written form are renamed to the
# title string first (data_processing/dataset/link_box_names.py), so the mention counts
# as depicted.
[ -d $U/xml_linked ] || python3 data_processing/dataset/link_box_names.py $U/xml $U/xml_linked
# Every log of the project lives under log/: training output in log/train,
# fitlog records in log/fitlog, and completion markers in log/run/done.

TASK=${1:-}; shift || true
ARGS=("$@")
FOLDS="0 1 2 3 4"; SEEDS="16 42 2023"; GPUS="1 2"; EPOCHS=""; PROPOSERS=""
CKPT=last; DRY=0
MODELS="gpt-6.1-sol gpt-6-luna gpt-5.5 gpt-5.4 gpt-4.1"
while [ $# -gt 0 ]; do
    case $1 in
        --folds) FOLDS=$2; shift 2;;
        --seeds) SEEDS=$2; shift 2;;
        --gpus) GPUS=$2; shift 2;;
        --epochs) EPOCHS=$2; shift 2;;
        --proposers) PROPOSERS=$2; shift 2;;
        --models) MODELS=$2; shift 2;;
        --ckpt) CKPT=$2; shift 2;;
        --dry) DRY=1; shift;;
        *) echo "unknown option $1" >&2; exit 1;;
    esac
done

if [ "$TASK" = all ]; then
    # every task in turn with the same options; a task that cannot run is skipped
    for T in nergl nollm proposers bert bertcrf maf hvpnet ita mkgformer llm; do
        bash "$0" $T "${ARGS[@]}" || echo "task $T did not finish"
    done
    exit 0
fi

feat_dir() {   # region features of one proposer
    case $1 in
        vinvl) echo $U/Ukiyoe1000_VinVL/;;
        gdino) echo $U/Ukiyoe1000_GDino_VL/;;           # Grounding DINO's own query vectors
        gdinop) echo $U/Ukiyoe1000_GDino_Prompt_VL/;;
        gdinores) echo $U/Ukiyoe1000_GDino/;;            # same boxes, ResNet-50 crop features
        gdinopres) echo $U/Ukiyoe1000_GDino_Prompt/;;
        florence) echo $U/Ukiyoe1000_Florence/;;
        rtdetr) echo $U/Ukiyoe1000_RTDETR/;;
        yolo) echo $U/Ukiyoe1000_Yolo/;;
        clip) echo $U/Ukiyoe1000_CLIP/;;
        *) echo "unknown proposer $1" >&2; exit 1;;
    esac
}

gdino_features() {
    # Grounding DINO scores shift by about 1e-3 between GPU models, which reorders
    # near-tied boxes, so extraction runs on GPU1, the RTX A6000 that produced the
    # boxes already on disk.
    local V D R
    for V in base prompt; do
        D=$U/Ukiyoe1000_GDino_VL/; R=$U/Ukiyoe1000_GDino/
        [ $V = prompt ] && { D=$U/Ukiyoe1000_GDino_Prompt_VL/; R=$U/Ukiyoe1000_GDino_Prompt/; }
        [ "$(ls $D 2>/dev/null | wc -l)" -ge 1078 ] && continue
        CUDA_VISIBLE_DEVICES=1 python3 data_processing/features/generate_gdino_vl_features.py \
            --variant $V --output_dir $D --ref_dir $R --device 0 > $LOG_DIR/run/gdino_vl_$V.log 2>&1
    done
}

# ---------------------------------------------------------------- one job each
job_nergl() {   # proposer fold seed gpu
    local P=$1 K=$2 S=$3 G=$4 NAME="${PREFIX}_${1}_e${EPOCHS}_f${2}_s${3}"
    [ -f "$LOG_DIR/run/done/$NAME" ] && { echo "skip $NAME"; return; }
    local SAVE=1 EVERY=$EPOCHS; [ $CKPT = none ] && SAVE=0
    echo "[$(date +%H:%M:%S)] START $NAME GPU$G"
    RAG_METHOD=$RAGM RAG_GPT_FILE=$SOL EARLY_STOP_PATIENCE=0 SAVE_EVERY=$EVERY \
    CUDA_VISIBLE_DEVICES=$G python3 script/train.py \
        --bart_name $BART --n_epochs $EPOCHS --seed $S \
        --datapath $U/cv/fold$K/ --image_feature_path $(feat_dir $P) \
        --image_annotation_path $U/xml_linked/ \
        --lr 3e-5 --box_num 18 --batch_size 16 --max_len 50 \
        --save_model $SAVE --normalize --use_kl \
        --save_path $SAVE_DIR/${NAME}_model \
        --region_loss_ratio 1.0 --log $LOG_DIR/fitlog/$NAME/ \
        > $LOG_DIR/train/train_$NAME.log 2>&1 \
      && touch $LOG_DIR/run/done/$NAME
    echo "[$(date +%H:%M:%S)] END $NAME"
}

has_baseline() {   # the code of one baseline, or skip the task
    [ -f "$BASELINE_DIR/$1" ] && return
    echo "skip $TASK: $BASELINE_DIR/$1 not found (third-party baseline, not in the repository)"
    exit 0
}

job_text() {    # bert|bertcrf fold seed gpu
    local T=$1 K=$2 S=$3 G=$4 NAME="${1}_f${2}_s${3}"
    local OUT=$SAVE_DIR/$NAME.json
    [ -f $OUT ] && { echo "skip $NAME"; return; }
    local CRF=""; [ $T = bertcrf ] && CRF=--crf
    echo "[$(date +%H:%M:%S)] START $NAME GPU$G"
    CUDA_VISIBLE_DEVICES=$G python3 $BASELINE_DIR/text/run_text_baselines.py \
        --epochs 10 --seed $S $CRF --txt_dir $U/cv/fold$K --out $OUT \
        > $LOG_DIR/train/train_$NAME.log 2>&1
    echo "[$(date +%H:%M:%S)] END $NAME"
}

job_mner() {    # method fold seed gpu; each port lives in $BASELINE_DIR/<method>/
    local M=$1 K=$2 S=$3 G=$4 NAME="${1}_f${2}_s${3}"
    [ -f $SAVE_DIR/$NAME.json ] && { echo "skip $NAME"; return; }
    echo "[$(date +%H:%M:%S)] START $NAME GPU$G"
    bash $BASELINE_DIR/$M/run_fold.sh $K $S $G   # writes $SAVE_DIR/$NAME.json and $LOG_DIR/train/train_$NAME.log
    echo "[$(date +%H:%M:%S)] END $NAME"
}

job_llm() {     # model
    local M=$1 OUT=$SAVE_DIR/llm_v2
    # Complete only if every print got an answer; a run cut short by the API
    # quota leaves empty answers behind and is redone.
    local R=$OUT/${M}_gmner_result.txt
    [ -f $R ] && ! grep -q "^  LLM: *$" $R && { echo "skip $M"; return; }
    echo "[$(date +%H:%M:%S)] START $M"
    python3 data_processing/evaluation/eval_llm_gmner.py \
        --model $M --test_txt $U/cv/all.txt --image_dir $IMAGE_DIR/ \
        --vinvl_dir $U/Ukiyoe1000_VinVL/ --xml_dir $U/xml_linked/ \
        --output_dir $OUT --concurrency 6 > $LOG_DIR/eval/eval_llm_$M.log 2>&1
    echo "[$(date +%H:%M:%S)] END $M"
}

# ---------------------------------------------------------------- job list
JOBS=()
case $TASK in
    nergl|proposers|nollm)
        PREFIX=nergl; RAGM=1
        [ $TASK = nollm ] && { PREFIX=nollm; RAGM=0; }   # RAG_METHOD 0: empty knowledge stream
        [ $TASK != proposers ] && { : ${EPOCHS:=50}; : ${PROPOSERS:=vinvl}; }
        [ $TASK = proposers ] && { : ${EPOCHS:=30}; : ${PROPOSERS:=all}; }
        [ "$PROPOSERS" = all ] && PROPOSERS="vinvl gdino gdinop florence rtdetr yolo clip"
        for P in $PROPOSERS; do feat_dir $P >/dev/null || exit 1; done
        HAVE=""
        for P in $PROPOSERS; do
            if [ -d $(feat_dir $P) ]; then HAVE="$HAVE $P"
            else echo "skip proposer $P: $(feat_dir $P) not found (python3 script/download.py --proposers)"; fi
        done
        PROPOSERS=$HAVE
        for P in $PROPOSERS; do for K in $FOLDS; do for S in $SEEDS; do
            JOBS+=("job_nergl $P $K $S"); done; done; done;;
    bert|bertcrf)
        has_baseline text/run_text_baselines.py
        for K in $FOLDS; do for S in $SEEDS; do JOBS+=("job_text $TASK $K $S"); done; done;;
    maf|hvpnet|ita|mkgformer)
        has_baseline $TASK/run_fold.sh
        for K in $FOLDS; do for S in $SEEDS; do JOBS+=("job_mner $TASK $K $S"); done; done;;
    llm)
        for M in $MODELS; do JOBS+=("job_llm $M"); done;;
    *) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 1;;
esac

echo "task $TASK: ${#JOBS[@]} jobs on GPUs [$GPUS]"
if [ $DRY = 1 ]; then printf '  %s\n' "${JOBS[@]}"; exit 0; fi

if [ $TASK = llm ]; then
    # Every print once: the five test folds together are the whole corpus.
    [ -f $U/cv/all.txt ] || cat $U/cv/fold{0,1,2,3,4}/test.txt > $U/cv/all.txt
    [ -f .env ] && { set -a; . ./.env; set +a; }
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "skip llm: OPENAI_API_KEY not set in .env"; exit 0; }
    # API calls, not GPU work: three models at a time.
    for i in "${!JOBS[@]}"; do
        ${JOBS[$i]} &
        [ $(( (i + 1) % 3 )) = 0 ] && wait
    done
    wait
else
    [[ " $PROPOSERS " =~ " gdino" ]] && gdino_features
    G=($GPUS)
    worker() {
        local w=$1 i
        for (( i = w; i < ${#JOBS[@]}; i += ${#G[@]} )); do ${JOBS[$i]} ${G[$w]}; done
    }
    for w in "${!G[@]}"; do worker $w & done
    wait
fi
echo "[$(date +%H:%M:%S)] $TASK COMPLETE"
