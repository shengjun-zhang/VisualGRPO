export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export MASTER_PORT=29513

torchrun --nnodes=1 --nproc_per_node=4 --node_rank 0 \
    fastvideo/train_g2rpo_rlpt_dino.py \
    --seed 42 \
    --pretrained_model_name_or_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/ckpt/flux \
    --hps_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/ckpt/hps/HPS_v2.1_compressed.pt \
    --hps_clip_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin \
    --dino_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/dinov2 \
    --data_json_path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/datasets/flux_rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --max_train_steps 301 \
    --learning_rate 2e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --cfg 0.0 \
    --output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/save_exp_rlpt/hps_gt \
    --h 1024 \
    --w 1024 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.7 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 12 \
    --shift 3 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --eta_step_list 0 1 2 3 4 5 6 7 \
    --granular_list 1 \