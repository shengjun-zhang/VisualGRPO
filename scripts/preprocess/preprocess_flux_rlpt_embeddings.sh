GPU_NUM=8 # 2,4,8
MODEL_PATH="./ckpt/flux"
OUTPUT_DIR="data/laion_rl_embeddings"  # Updated for LAION dataset
PROMPT_DIR="/data2/dataset/laion-220k/short_captions.txt"  # Path to LAION captions

echo "[INFO] Processing LAION-220k dataset captions..."
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Prompt file: $PROMPT_DIR"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding_rlpt.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir $PROMPT_DIR