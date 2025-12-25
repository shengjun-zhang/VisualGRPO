#!/usr/bin/env python
# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
VisualGRPO 训练入口。

使用 Hydra 进行配置管理，支持多种 RL 算法和奖励模型。

用法:
    # 使用默认配置
    torchrun --nproc_per_node=8 train.py
    
    # 覆盖配置
    torchrun --nproc_per_node=8 train.py \
        model.pretrained_model_name_or_path=/path/to/flux \
        reward.hps_path=/path/to/hps.pt \
        training.max_train_steps=500
    
    # 使用不同的配置组合
    torchrun --nproc_per_node=8 train.py \
        reward=clip \
        algorithm=grpo
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate.utils import set_seed
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.optimization import get_scheduler

from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.logging_ import main_print
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
from fastvideo.trainer import GRPOTrainer
from fastvideo.workers import RewardManager


def init_distributed():
    """初始化分布式环境。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 先设置 CUDA 设备
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    if world_size > 1:
        # 指定 device_id 避免 NCCL hang
        dist.init_process_group(
            "nccl",
            device_id=torch.device(f"cuda:{local_rank}")
        )
    
    return local_rank, rank, world_size, device


def build_model(cfg: DictConfig, device: torch.device):
    """构建 Diffusion 模型。"""
    main_print(f"--> Loading model from {cfg.model.pretrained_model_name_or_path}")
    
    # 加载 transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    
    # 应用 FSDP
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        cfg.distributed.fsdp_sharding_strategy,
        False,  # use_lora
        cfg.distributed.use_cpu_offload,
        cfg.distributed.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs)
    
    # 应用 gradient checkpointing
    if cfg.training.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, cfg.training.selective_checkpointing
        )
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    main_print(f"--> Model loaded, FSDP strategy: {cfg.distributed.fsdp_sharding_strategy}")
    
    return transformer, vae


def build_reward_manager(cfg: DictConfig, device: torch.device) -> RewardManager:
    """构建奖励管理器。"""
    reward_config = {
        "type": cfg.reward.reward_type,
        "model_name": cfg.reward.get("model_name", "ViT-H-14"),
        "hps_path": cfg.reward.get("hps_path"),
        "clip_path": cfg.reward.get("clip_path"),
        "pretrained": cfg.reward.get("pretrained"),
        "weight": cfg.reward.get("weight", 1.0),
    }
    
    # 如果是组合奖励
    if cfg.reward.reward_type == "combined":
        reward_config["models"] = OmegaConf.to_container(cfg.reward.models)
    
    return RewardManager(device=device, reward_config=reward_config)


def build_dataloader(cfg: DictConfig, rank: int, world_size: int):
    """构建数据加载器。"""
    train_dataset = LatentDataset(
        cfg.data.json_path, 
        cfg.data.num_latent_t, 
        cfg.data.cfg_rate
    )
    
    sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True, 
        seed=cfg.sampling.sampler_seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=cfg.data.train_batch_size,
        num_workers=cfg.data.dataloader_num_workers,
        drop_last=True,
    )
    
    return train_dataloader, sampler


def build_optimizer_and_scheduler(cfg: DictConfig, transformer):
    """构建优化器和学习率调度器。"""
    params_to_optimize = list(filter(
        lambda p: p.requires_grad, 
        transformer.parameters()
    ))
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
        eps=1e-8,
    )
    
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=cfg.training.lr_num_cycles,
        power=cfg.training.lr_power,
    )
    
    return optimizer, lr_scheduler


def build_trainer_config(cfg: DictConfig) -> dict:
    """从 Hydra 配置构建 Trainer 配置字典。"""
    return {
        # 采样配置
        "height": cfg.sampling.height,
        "width": cfg.sampling.width,
        "sampling_steps": cfg.sampling.sampling_steps,
        "eta": cfg.sampling.eta,
        "shift": cfg.sampling.shift,
        "guidance_scale": cfg.sampling.guidance_scale,
        
        # GRPO 配置
        "num_generations": cfg.grpo.num_generations,
        "init_same_noise": cfg.grpo.init_same_noise,
        "clip_range": cfg.grpo.clip_range,
        "adv_clip_max": cfg.grpo.adv_clip_max,
        "eta_step_list": list(cfg.grpo.eta_step_list),
        "eta_step_merge_list": list(cfg.grpo.eta_step_merge_list),
        "granular_list": list(cfg.grpo.granular_list),
        
        # 训练配置
        "max_grad_norm": cfg.training.max_grad_norm,
        "output_dir": cfg.output_dir,
        "checkpointing_steps": cfg.training.checkpointing_steps,
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """主训练函数。"""
    # 打印配置
    main_print(OmegaConf.to_yaml(cfg))
    
    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 初始化分布式
    local_rank, rank, world_size, device = init_distributed()
    
    # 初始化序列并行
    initialize_sequence_parallel_state(cfg.distributed.sp_size)
    
    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed + rank)
    
    # 创建输出目录
    if rank == 0 and cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 构建模型
    transformer, vae = build_model(cfg, device)
    transformer.train()
    
    # 构建奖励管理器
    reward_manager = build_reward_manager(cfg, device)
    
    # 构建数据加载器
    train_dataloader, sampler = build_dataloader(cfg, rank, world_size)
    
    # 构建优化器和调度器
    optimizer, lr_scheduler = build_optimizer_and_scheduler(cfg, transformer)
    
    # 创建数据加载器包装器
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        cfg.data.train_batch_size,
        cfg.distributed.sp_size,
        cfg.distributed.train_sp_batch_size,
    )
    
    # 构建 Trainer 配置
    trainer_config = build_trainer_config(cfg)
    
    # 关键：在训练开始前同步所有进程，确保所有模型都加载完成
    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        main_print("--> All processes synchronized, ready to start training")
    
    # 打印训练信息
    total_batch_size = (
        cfg.data.train_batch_size 
        * world_size 
        / cfg.distributed.sp_size
        * cfg.distributed.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataloader.dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Total train batch size = {total_batch_size}")
    main_print(f"  Total optimization steps = {cfg.training.max_train_steps}")
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")
    
    # 构建 WandB 配置
    wandb_config = None
    if hasattr(cfg, 'wandb'):
        wandb_config = OmegaConf.to_container(cfg.wandb)
    
    # 创建 Trainer
    trainer = GRPOTrainer(
        config=trainer_config,
        transformer=transformer,
        vae=vae,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        data_loader_wrapper=loader,
        device=device,
        reward_manager=reward_manager,
        wandb_config=wandb_config,
    )
    
    # 开始训练
    trainer.train(num_steps=cfg.training.max_train_steps)
    
    # 清理
    trainer.finish()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    
    main_print("Training completed!")


if __name__ == "__main__":
    main()
