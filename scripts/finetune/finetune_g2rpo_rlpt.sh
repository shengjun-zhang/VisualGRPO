#!/bin/bash

# LAION-220k Image-Text RL Training Configuration
echo "[INFO] Starting LAION-220k Image-Text RL Training..."
echo "[INFO] Using GT images from /data2/dataset/laion-220k/images"
echo "[INFO] Using embeddings from data/laion_rl_embeddings/videos2caption.json"

torchrun --nproc_per_node=8 --master_port 11451 \
    fastvideo/train_grpo_rlpt.py \
    --seed 42 \
    --pretrained_model_name_or_path ckpt/flux \
    --hps_path ckpt/hps/HPS_v2.1_compressed.pt \
    --hps_clip_path ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin \
    --clip_score_path  ckpt/clip_score \
    --data_json_path data/laion_rl_embeddings/videos2caption.json \
    --image_data_dir /data2/dataset/laion-220k/images \
    --log_file save_exp/laion_hps_clip_mse/training_logs.csv \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --max_train_steps 301 \
    --learning_rate 2e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 10 \
    --cfg 0.0 \
    --output_dir save_exp/laion_hps_clip_mse \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.7 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --shift 3 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --eta_step_list 0 1 2 3 \
    --granular_list 1 \
    --use_hps_reward \
    --use_clip_reward \
    --use_mse_reward \
    --hps_reward_weight 1.0 \
    --clip_reward_weight 1.0 \
    --mse_reward_weight 1.0