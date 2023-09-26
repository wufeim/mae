python submitit_pretrain.py \
    --job_name mae_small_in100_epo800_8node_8gpu \
    --job_dir /data/home/wufeim/dst/mae/exp/outputs_mae_small_in100_epo800_8node_8gpu \
    --partition learnai \
    --timeout 72 \
    --nodes 8 \
    --batch_size 64 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /fsx/wufeim/IN100

exit
