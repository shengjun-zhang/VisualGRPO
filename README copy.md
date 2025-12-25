# E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models

**[E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models]()**
</br>
[Shengjun Zhang](),
[Zhang Zhang](),
[Chensheng Dai](),
[Yueqi Duan<sup>†</sup>]()
(<sup>†</sup>Corresponding Author)

[![arXiv]()]()
[![Project Page]()]()
[![Hugging Face]()]()

## 📜 News

**[2025/12/24]** Code is available now!

## 🚀 Method Overview

<div align="center">
    <img src=''/>
</div>

E-GRPO (Entropy-Guided GRPO) is a novel reinforcement learning approach for flow-based diffusion models. Our key insight is that **high-entropy denoising steps are more critical for policy optimization**, and we propose a merging-step strategy that focuses training on these important steps.

### Key Features

- **E-GRPO Algorithm**: Novel merging-step strategy focusing on high-entropy timesteps
- **Multi-Granularity Rewards**: Support for computing rewards at multiple sampling granularities
- **Flexible Architecture**: Modular design supporting multiple RL algorithms and reward models
- **Distributed Training**: Full support for FSDP and sequence parallelism

### Supported Algorithms

This repository not only implements E-GRPO but also provides a unified framework for various GRPO variants:

| Algorithm | Description | Config |
|-----------|-------------|--------|
| **E-GRPO (grpo_merge)** | Our method - merging-step strategy | `algorithm=grpo_merge` |
| DanceGRPO | Basic full-step SDE sampling | `algorithm=dance_grpo` |
| DanceGRPO LoRA | LoRA fine-tuning variant | `algorithm=dance_grpo_lora` |
| MixGRPO | Mixed SDE-ODE sampling with sliding window | `algorithm=mix_grpo` |
| BranchGRPO | Tree-based sampling with split and pruning | `algorithm=branch_grpo` |

### Supported Reward Models

| Model | Description | Config |
|-------|-------------|--------|
| **HPS v2** | Human Preference Score | `reward=hps` |
| CLIP Score | Text-image alignment | `reward=clip` |
| ImageReward | Learned human preference | `reward=image_reward` |
| PickScore | Pick-a-pic preference model | `reward=pick_score` |
| Multi-Reward | Combination of multiple rewards | `reward=multi_reward` |

## 🔧 Installation

### Setup Repository and Conda Environment

```bash
git clone https://github.com/your-repo/VisualRL.git
cd VisualRL

# Create conda environment
conda create -n e-grpo python=3.10 -y
conda activate e-grpo

# Install dependencies
pip install -e .
```

The environment dependency is compatible with [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

## 🔑 Model Preparations

### 1. FLUX Model

```bash
# Download FLUX model from Hugging Face
mkdir -p ckpt/flux
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ckpt/flux
```

### 2. Reward Models

#### HPS v2.1

```bash
# Clone HPSv2 repository
git clone https://github.com/tgxs002/HPSv2.git

# Download HPS checkpoint
mkdir -p ckpt/hps
wget -O ckpt/hps/HPS_v2_compressed.pt https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt

# Download CLIP model
mkdir -p ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K
```

## 📊 Data Preparation

### 1. Prepare Text Prompts

Create a text file with your prompts (one per line):

```bash
# Example: prompts.txt
A photo of a cat sitting on a windowsill
A beautiful sunset over the ocean
...
```

### 2. Preprocess Text Embeddings

```bash
bash scripts/preprocess/preprocess_prompts.sh
```

### 3. Generate Index File

```bash
python scripts/preprocess/generate_json_index.py \
    --embedding_dir data/rl_embeddings \
    --output_path data/rl_embeddings/videos2caption.json
```

## 🎈 Quick Start

### Train E-GRPO (Our Method)

```bash
# Using HPS reward
bash scripts/finetune/train_grpo_hps.sh

# Or with custom settings
torchrun --nproc_per_node=8 fastvideo/train.py \
    algorithm=grpo_merge \
    reward=hps \
    model.pretrained_model_name_or_path=./ckpt/flux \
    data.json_path=./data/rl_embeddings/videos2caption.json \
    training.max_train_steps=300
```

### Train Other Algorithms

```bash
# DanceGRPO (basic GRPO)
bash scripts/finetune/train_dance_grpo.sh

# MixGRPO (mixed SDE-ODE)
bash scripts/finetune/train_mix_grpo.sh

# BranchGRPO (tree-based sampling)
bash scripts/finetune/train_branch_grpo.sh

# Multi-reward training
bash scripts/finetune/train_multi_reward.sh
```

### Configuration

All configurations are managed by [Hydra](https://hydra.cc/). You can override any config value from command line:

```bash
torchrun --nproc_per_node=8 fastvideo/train.py \
    algorithm=grpo_merge \
    reward=hps \
    training.learning_rate=1e-6 \
    training.max_train_steps=500 \
    grpo.num_generations=16 \
    sampling.height=512 \
    sampling.width=512
```

## 📁 Project Structure

```
VisualRL/
├── fastvideo/                    # Main source code
│   ├── config/                   # Hydra configurations
│   │   ├── algorithm/            # Algorithm configs (grpo_merge, dance_grpo, etc.)
│   │   ├── reward/               # Reward model configs (hps, clip, etc.)
│   │   ├── model/                # Model configs
│   │   └── config.yaml           # Main config
│   ├── trainer/                  # Training logic
│   │   ├── base_trainer.py       # Abstract trainer base
│   │   └── grpo_trainer.py       # GRPO trainer implementation
│   ├── workers/                  # Sampling and reward components
│   │   ├── sampler/              # Sampling strategies
│   │   ├── reward_manager/       # Reward model management
│   │   └── rollout_buffer.py     # Rollout data storage
│   ├── train.py                  # Training entry point
│   └── ...
├── scripts/                      # Shell scripts
│   ├── finetune/                 # Training scripts
│   └── preprocess/               # Data preprocessing scripts
├── docs/                         # Documentation
└── other_algorithms/             # Reference implementations
```

## 📎 Citation 

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@article{zhang2025egrpo,
  title={E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models},
  author={Zhang, Shengjun and Zhang, Zhang and Dai, Chensheng and Duan, Yueqi},
  journal={arXiv preprint},
  year={2025}
}
```

## 💞 Acknowledgement

This codebase is built upon the following excellent repositories:

* [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) - Basic GRPO implementation
* [Flow-GRPO](https://github.com/yifan123/flow_grpo) - Flow-based GRPO
* [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO) - Mixed sampling strategy
* [Granular-GRPO](https://github.com/bcmi/Granular-GRPO) - Multi-granularity rewards
* [BranchGRPO](https://github.com/Fredreic1849/BranchGRPO) - Tree-based sampling
* [FastVideo](https://github.com/hao-ai-lab/FastVideo) - Distributed training framework
* [DDPO](https://github.com/kvablack/ddpo-pytorch) - Diffusion policy optimization

## 📄 License

This project is licensed under the Apache License 2.0.
