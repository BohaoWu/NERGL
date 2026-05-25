
CUDA_VISIBLE_DEVICES=1 python test.py \
    --bart_name /root/GMNER/download_model/bart-large-japanese \
    --model_weight ./saved_model/best_model \
    --datapath  ./Ukiyoe1000/txt \
    --image_feature_path ./Ukiyoe1000_VinVL \
    --image_annotation_path ./Ukiyoe1000/xml \
    --box_num 18 \
    --batch_size 32 \
    --max_len 50 \
    --normalize \
          