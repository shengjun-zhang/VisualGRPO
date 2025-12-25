#!/bin/bash
# VisualGRPO 数据预处理脚本
# 将 prompts.txt 预处理为训练所需的 embeddings

# 配置路径
FLUX_MODEL_PATH="/data2/zsj/VisualGRPO/VisualRL/ckpt/flux"
PROMPT_FILE="/data2/zsj/VisualGRPO/VisualRL/prompts.txt"
OUTPUT_DIR="/data2/zsj/VisualGRPO/VisualRL/data/rl_embeddings"

# 分布式配置（使用多 GPU 加速预处理）
NUM_GPUS=8

# 运行预处理
torchrun --nproc_per_node=$NUM_GPUS \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $FLUX_MODEL_PATH \
    --prompt_dir $PROMPT_FILE \
    --output_dir $OUTPUT_DIR \
    --train_batch_size 1 \
    --dataloader_num_workers 4

echo "预处理完成！输出目录: $OUTPUT_DIR"
echo "生成的文件:"
echo "  - $OUTPUT_DIR/videos2caption.json (训练用的索引文件)"
echo "  - $OUTPUT_DIR/prompt_embed/*.pt (prompt embeddings)"
echo "  - $OUTPUT_DIR/pooled_prompt_embeds/*.pt (pooled embeddings)"
echo "  - $OUTPUT_DIR/text_ids/*.pt (text IDs)"

