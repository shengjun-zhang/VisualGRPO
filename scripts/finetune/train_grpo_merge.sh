#!/bin/bash
# VisualGRPO Merging Step 训练脚本示例
# 使用 HPS 奖励模型 + Merging Step GRPO

# 配置路径
MODEL_PATH="/path/to/flux"
HPS_PATH="/path/to/HPS_v2_compressed.pt"
HPS_CLIP_PATH="/path/to/open_clip_pytorch_model.bin"
DATA_PATH="/path/to/data/rl_embeddings/videos2caption.json"
OUTPUT_DIR="exp_flux/grpo_merge"

# 分布式配置
NUM_GPUS=8

# 使用 grpo_merge 算法配置
torchrun --nproc_per_node=$NUM_GPUS \
    fastvideo/train.py \
    algorithm=grpo_merge \
    model.pretrained_model_name_or_path=$MODEL_PATH \
    reward.hps_path=$HPS_PATH \
    reward.clip_path=$HPS_CLIP_PATH \
    data.json_path=$DATA_PATH \
    output_dir=$OUTPUT_DIR \
    training.max_train_steps=300 \
    grpo.eta_step_list=[1,2,3] \
    grpo.eta_step_merge_list=[1,1,1] \
    grpo.granular_list=[1,2]


