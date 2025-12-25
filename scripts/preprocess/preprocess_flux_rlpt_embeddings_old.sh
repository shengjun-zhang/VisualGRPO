GPU_NUM=4 # 2,4,8
MODEL_PATH="./ckpt/flux"
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/checkpoints/G2RPO/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./prompts.txt"