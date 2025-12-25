#!/bin/bash
# VisualGRPO 训练脚本
# 使用 HPS 奖励模型进行 GRPO 训练

# 确保在项目根目录运行
cd /data2/zsj/VisualGRPO/VisualRL

# 分布式配置
NUM_GPUS=8

# 启动训练（使用默认配置，模型路径已在 config 中设置）
torchrun --nproc_per_node=$NUM_GPUS \
    fastvideo/train.py \
    output_dir=exp_flux/grpo_hps \
    training.max_train_steps=300 \
    training.learning_rate=1e-5 \
    grpo.num_generations=12

# 如果需要启用 WandB，添加：
# wandb.enabled=true \
# wandb.project=VisualGRPO \
# wandb.name=grpo_hps_run1
