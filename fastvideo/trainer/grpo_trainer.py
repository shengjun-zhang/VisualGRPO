# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
GRPOTrainer: GRPO 算法的训练器实现。

整合了采样、奖励计算、优势计算和策略更新的完整流程。
支持：
- Merging Step GRPO
- 多粒度奖励
- FSDP 分布式训练
- 混合精度训练
- WandB 日志记录
"""

import os
import time
from collections import deque
from typing import Dict, Any, Optional, Iterator

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from fastvideo.workers import FluxSampler, RewardManager, RolloutBuffer
from fastvideo.utils.checkpoint import save_checkpoint
from fastvideo.utils.logging_ import main_print

# WandB 支持（可选）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程间收集 tensor。"""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


class GRPOTrainer(BaseTrainer):
    """
    GRPO 训练器实现。
    
    训练流程：
    1. 从 DataLoader 获取 batch
    2. 使用 Sampler 进行 rollout，得到 RolloutBuffer
    3. 使用 RewardManager 计算奖励
    4. 计算 GRPO 优势值
    5. 执行 PPO-style 策略更新
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        data_loader_wrapper: Iterator,
        device: torch.device,
        reward_manager: Optional[RewardManager] = None,
        sampler: Optional[FluxSampler] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 GRPO 训练器。
        
        Args:
            config: 训练配置
            transformer: Diffusion transformer 模型
            vae: VAE 模型
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            train_dataloader: 训练数据加载器
            data_loader_wrapper: 数据加载器包装器（用于序列并行）
            device: 计算设备
            reward_manager: 奖励管理器
            sampler: 采样器
            wandb_config: WandB 配置（可选）
        """
        super().__init__(
            config, transformer, optimizer, lr_scheduler, 
            train_dataloader, device
        )
        
        self.vae = vae
        self.data_loader_wrapper = data_loader_wrapper
        self.reward_manager = reward_manager
        
        # GRPO 配置
        self.num_generations = config.get("num_generations", 12)
        self.clip_range = config.get("clip_range", 1e-4)
        self.adv_clip_max = config.get("adv_clip_max", 5.0)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.eta = config.get("eta", 0.7)
        self.eta_step_list = config.get("eta_step_list", [1])
        self.eta_step_merge_list = config.get("eta_step_merge_list", [1])
        
        # 采样器配置
        sampler_config = {
            "height": config.get("height", 720),
            "width": config.get("width", 720),
            "sampling_steps": config.get("sampling_steps", 16),
            "shift": config.get("shift", 3.0),
            "eta": self.eta,
            "guidance_scale": config.get("guidance_scale", 3.5),
            "num_generations": self.num_generations,
            "init_same_noise": config.get("init_same_noise", True),
            "eta_step_list": self.eta_step_list,
            "eta_step_merge_list": self.eta_step_merge_list,
            "granular_list": config.get("granular_list", [1, 2]),
        }
        
        # 初始化采样器
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = FluxSampler(
                transformer=transformer,
                vae=vae,
                device=device,
                config=sampler_config,
            )
        
        # 训练统计
        self.step_times = deque(maxlen=100)
        
        # 输出目录
        self.output_dir = config.get("output_dir", "exp_flux/test")
        self.checkpointing_steps = config.get("checkpointing_steps", 500)
        
        # 分布式信息
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # WandB 配置
        self.use_wandb = False
        if wandb_config is not None and wandb_config.get("enabled", False):
            if WANDB_AVAILABLE and self.rank == 0:
                self.use_wandb = True
                self._init_wandb(wandb_config, config)
    
    def _init_wandb(self, wandb_config: Dict[str, Any], train_config: Dict[str, Any]):
        """初始化 WandB。"""
        wandb.init(
            project=wandb_config.get("project", "VisualGRPO"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=train_config,
            dir=self.output_dir,
        )
        main_print(f"WandB initialized: {wandb.run.name}")
    
    def _log_to_wandb(self, metrics: Dict[str, Any], step: int):
        """记录指标到 WandB。"""
        if self.use_wandb and self.rank == 0:
            wandb.log(metrics, step=step)
    
    def _repeat_for_generations(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """将 tensor 重复 num_generations 次。"""
        if tensor is None:
            return None
        return torch.repeat_interleave(tensor, self.num_generations, dim=0)
    
    def _prepare_batch(self, batch):
        """准备训练批次数据。"""
        encoder_hidden_states, pooled_prompt_embeds, text_ids, captions = batch
        
        # 重复数据用于 GRPO
        encoder_hidden_states = self._repeat_for_generations(encoder_hidden_states)
        pooled_prompt_embeds = self._repeat_for_generations(pooled_prompt_embeds)
        text_ids = self._repeat_for_generations(text_ids)
        
        # 处理 captions
        if isinstance(captions, str):
            captions = [captions] * self.num_generations
        elif isinstance(captions, (list, tuple)):
            captions = [item for item in captions for _ in range(self.num_generations)]
        
        return encoder_hidden_states, pooled_prompt_embeds, text_ids, captions
    
    def _sample_and_compute_rewards(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        captions: list,
    ) -> RolloutBuffer:
        """执行采样并计算奖励。"""
        rollout_buffer = self.sampler.sample(
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            captions=captions,
            reward_manager=self.reward_manager,
        )
        
        # 计算优势值
        rollout_buffer.compute_advantages(
            num_generations=self.num_generations,
            normalize=True,
        )
        
        return rollout_buffer
    
    def _compute_grpo_loss(
        self,
        pred: torch.Tensor,
        rollout_buffer: RolloutBuffer,
        step_idx: int,
    ) -> torch.Tensor:
        """计算 GRPO 损失。"""
        # 获取当前步的数据
        batch = rollout_buffer.get_training_batch(step_idx)
        
        # 计算新的 log probability
        new_log_probs = self.sampler.compute_log_prob(
            pred=pred,
            latents=batch["latents"],
            next_latents=batch["next_latents"],
            sigma_schedule=rollout_buffer.sigma_schedule,
            eta_step=self.eta_step_list[step_idx],
            merge_step=self.eta_step_merge_list[step_idx],
        )
        
        # 获取优势值并裁剪
        advantages = torch.clamp(
            batch["advantages"], -self.adv_clip_max, self.adv_clip_max
        )
        
        # 计算 ratio
        old_log_probs = rollout_buffer.log_probs[:, step_idx]
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO-style 裁剪损失
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        return loss, ratio, advantages
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        执行单步训练。
        
        Args:
            batch: 数据批次
            
        Returns:
            包含损失和其他指标的字典
        """
        self.optimizer.zero_grad()
        
        # 准备数据
        encoder_hidden_states, pooled_prompt_embeds, text_ids, captions = self._prepare_batch(batch)
        
        # 采样并计算奖励
        rollout_buffer = self._sample_and_compute_rewards(
            encoder_hidden_states, pooled_prompt_embeds, text_ids, captions
        )
        
        # 记录评估奖励
        eval_reward_mean = None
        if rollout_buffer.eval_rewards is not None:
            gathered_reward = gather_tensor(rollout_buffer.eval_rewards)
            eval_reward_mean = gathered_reward.mean().item()
            if self.rank == 0:
                main_print(f"gathered_hps_reward: {gathered_reward}")
                main_print(f"mean_hps_reward: {eval_reward_mean:.4f}")
                reward_path = os.path.join(self.output_dir, "hps_reward.txt")
                with open(reward_path, 'a') as f:
                    f.write(f"{eval_reward_mean}\n")
        
        total_loss = 0.0
        train_timesteps = rollout_buffer.num_steps
        
        # 对每个时间步进行训练
        for t_idx in range(train_timesteps):
            # 获取第一个样本的输入用于共享计算
            lat_0 = rollout_buffer.latents[0, t_idx].unsqueeze(0)
            t_0 = rollout_buffer.timesteps[0, t_idx].unsqueeze(0)
            enc_0 = rollout_buffer.encoder_hidden_states[0].unsqueeze(0)
            pooled_0 = rollout_buffer.pooled_prompt_embeds[0].unsqueeze(0)
            text_0 = rollout_buffer.text_ids[0].unsqueeze(0)
            image_0 = rollout_buffer.image_ids[0].unsqueeze(0)
            
            # 获取模型预测（训练模式）
            pred = self.sampler.get_pred_for_training(
                lat_0, enc_0, pooled_0, text_0, image_0, t_0
            )
            
            # 重复预测用于所有生成
            pred_batch = pred.repeat(self.num_generations, 1, 1)
            
            # 计算 GRPO 损失
            loss, ratio, advantages = self._compute_grpo_loss(
                pred_batch, rollout_buffer, t_idx
            )
            
            # 反向传播
            loss.backward()
            
            # 同步损失
            avg_loss = loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            
            # 梯度裁剪和优化
            if isinstance(self.transformer, FSDP):
                grad_norm = self.transformer.clip_grad_norm_(self.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(), self.max_grad_norm
                )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # 日志输出
            if self.rank % 8 == 0:
                main_print(f"ratio: {ratio}")
                main_print(f"advantage: {advantages}")
                main_print(f"loss: {loss.item()}")
            
            if dist.is_initialized():
                dist.barrier()
        
        metrics = {
            "loss": total_loss,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
        
        # 添加奖励指标
        if eval_reward_mean is not None:
            metrics["eval_reward"] = eval_reward_mean
        
        return metrics
    
    def train(self, num_steps: int) -> None:
        """
        执行完整训练。
        
        Args:
            num_steps: 训练步数
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        progress_bar = tqdm(
            range(1, num_steps + 1),
            initial=self.global_step,
            desc="Training",
            disable=self.local_rank > 0,
        )
        
        for step in progress_bar:
            start_time = time.time()
            
            # 保存检查点
            if step % self.checkpointing_steps == 0:
                self._save_checkpoint(step)
            
            # 获取数据
            batch = next(self.data_loader_wrapper)
            
            # 训练一步
            metrics = self.train_step(batch)
            
            # 更新统计
            step_time = time.time() - start_time
            self.step_times.append(step_time)
            
            # 更新进度条
            postfix = {
                "loss": f"{metrics['loss']:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": f"{metrics['grad_norm']:.4f}",
            }
            if "eval_reward" in metrics:
                postfix["reward"] = f"{metrics['eval_reward']:.4f}"
            progress_bar.set_postfix(postfix)
            
            # 记录到 WandB
            wandb_metrics = {
                "train/loss": metrics["loss"],
                "train/grad_norm": metrics["grad_norm"],
                "train/step_time": step_time,
                "train/lr": self.lr_scheduler.get_last_lr()[0],
            }
            if "eval_reward" in metrics:
                wandb_metrics["eval/reward"] = metrics["eval_reward"]
            self._log_to_wandb(wandb_metrics, step)
            
            self.global_step = step
    
    def _save_checkpoint(self, step: int) -> None:
        """保存检查点。"""
        ckpt_path = os.path.join(self.output_dir, "ckpt")
        save_checkpoint(self.transformer, self.rank, ckpt_path, step, self.epoch)
        
        if dist.is_initialized():
            dist.barrier()
    
    def finish(self):
        """训练结束时的清理工作。"""
        if self.use_wandb and self.rank == 0:
            wandb.finish()
