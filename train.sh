# for se in '42' '16' '34' '2023' '25' '23' '2022'
# --bart_name /root/GMNER/download_model/bart-large-japanese \
for se in '42'
do
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --bart_name /root/GMNER/download_model/bart-large-japanese \
    --n_epochs 30 \
    --seed 42 \
    --datapath  /root/GMNER/Ukiyoe1000/txt/ \
    --image_feature_path /root/GMNER/Ukiyoe1000_VinVL/ \
    --image_annotation_path /root/GMNER/Ukiyoe1000/xml/ \
    --lr 3e-5 \
    --box_num 18 \
    --batch_size 16 \
    --max_len 50 \
    --save_model 1 \
    --normalize \
    --use_kl \
    --save_path ./saved_model/best_model \
    --region_loss_ratio 1.0 \
    --log ./logs/
done