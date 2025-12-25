#!/usr/bin/env python
# TODO: Delete this file
"""
生成训练用的 JSON 索引文件。

当数据预处理完成但 JSON 文件没有生成时，可以使用这个脚本手动生成。
"""

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/data2/zsj/VisualGRPO/VisualRL/data/rl_embeddings",
        help="数据目录路径"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data2/zsj/VisualGRPO/VisualRL/prompts.txt",
        help="原始 prompt 文件路径"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="videos2caption.json",
        help="输出 JSON 文件名"
    )
    args = parser.parse_args()
    
    # 读取原始 prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # 检查已生成的 embedding 文件
    prompt_embed_dir = os.path.join(args.data_dir, "prompt_embed")
    existing_files = set(os.listdir(prompt_embed_dir))
    
    # 生成 JSON 数据
    json_data = []
    for filename in sorted(existing_files):
        if not filename.endswith(".pt"):
            continue
        
        idx = int(filename.replace(".pt", ""))
        
        # 检查所有必要的文件是否存在
        prompt_embed_path = os.path.join(args.data_dir, "prompt_embed", filename)
        pooled_path = os.path.join(args.data_dir, "pooled_prompt_embeds", filename)
        text_ids_path = os.path.join(args.data_dir, "text_ids", filename)
        
        if not all(os.path.exists(p) for p in [prompt_embed_path, pooled_path, text_ids_path]):
            print(f"Warning: Missing files for index {idx}, skipping...")
            continue
        
        # 获取对应的 caption
        if idx < len(prompts):
            caption = prompts[idx]
        else:
            print(f"Warning: Index {idx} out of range, using empty caption")
            caption = ""
        
        item = {
            "prompt_embed_path": filename,
            "pooled_prompt_embeds_path": filename,
            "text_ids": filename,
            "caption": caption,
            "length": 1  # 图片长度为 1
        }
        json_data.append(item)
    
    # 保存 JSON 文件
    output_path = os.path.join(args.data_dir, args.output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"生成完成！")
    print(f"  总样本数: {len(json_data)}")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    main()



