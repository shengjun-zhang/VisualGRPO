# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
RolloutBuffer: 结构化存储采样过程中产生的数据。

在 GRPO 训练中，Sampler 负责采样并返回 RolloutBuffer 对象，
Trainer 从 RolloutBuffer 中读取数据进行策略更新。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class RolloutBuffer:
    """
    存储单次 rollout 产生的所有数据。
    
    Attributes:
        latents: 输入 latents，形状 [B, num_steps, ...]
        next_latents: 输出 latents，形状 [B, num_steps, ...]
        log_probs: 采样时的 log probabilities，形状 [B, num_steps]
        rewards: 奖励值，形状 [B, num_steps] 或 List[Tensor] 用于多粒度奖励
        advantages: 优势值，形状 [B, num_steps]
        timesteps: 时间步，形状 [B, num_steps]
        
        encoder_hidden_states: 文本编码，形状 [B, L, D]
        pooled_prompt_embeds: 池化的 prompt embeddings，形状 [B, D]
        text_ids: 文本 IDs
        image_ids: 图像 IDs
        captions: 原始文本 captions
        
        eval_rewards: 评估用的奖励值（anchor sample 的奖励）
        sigma_schedule: sigma 调度表
    """
    
    # 核心采样数据
    latents: Optional[torch.Tensor] = None
    next_latents: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None
    
    # 奖励相关
    rewards: Optional[List[torch.Tensor]] = None  # 多粒度奖励
    advantages: Optional[torch.Tensor] = None
    eval_rewards: Optional[torch.Tensor] = None
    
    # 条件信息
    encoder_hidden_states: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    image_ids: Optional[torch.Tensor] = None
    captions: Optional[List[str]] = None
    
    # 调度相关
    sigma_schedule: Optional[torch.Tensor] = None
    
    # 元信息
    device: Optional[torch.device] = None
    batch_size: int = 0
    num_steps: int = 0
    
    def to(self, device: torch.device) -> "RolloutBuffer":
        """将所有 tensor 移动到指定设备。"""
        self.device = device
        
        tensor_attrs = [
            'latents', 'next_latents', 'log_probs', 'timesteps',
            'advantages', 'eval_rewards',
            'encoder_hidden_states', 'pooled_prompt_embeds', 
            'text_ids', 'image_ids', 'sigma_schedule'
        ]
        
        for attr in tensor_attrs:
            val = getattr(self, attr)
            if val is not None and isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        
        # 处理 rewards 列表
        if self.rewards is not None:
            self.rewards = [r.to(device) for r in self.rewards]
        
        return self
    
    def compute_advantages(
        self, 
        num_generations: int,
        normalize: bool = True,
    ) -> None:
        """
        计算 GRPO 风格的优势值。
        
        对于每个 prompt group，计算组内的归一化优势：
        advantage = (reward - group_mean) / group_std
        
        Args:
            num_generations: 每个 prompt 生成的样本数
            normalize: 是否进行归一化
        """
        if self.rewards is None or len(self.rewards) == 0:
            raise ValueError("Rewards not set, cannot compute advantages")
        
        # 使用第一个粒度的奖励作为基准形状
        base_reward = self.rewards[0]
        n_prompts = base_reward.shape[0] // num_generations
        
        advantages = torch.zeros_like(base_reward)
        
        # 累加所有粒度的优势
        for reward in self.rewards:
            group_advantages = torch.zeros_like(reward)
            
            for i in range(n_prompts):
                start_idx = i * num_generations
                end_idx = (i + 1) * num_generations
                
                group_rewards = reward[start_idx:end_idx]
                group_mean = group_rewards.mean(dim=0)
                
                if normalize:
                    group_std = group_rewards.std(dim=0) + 1e-8
                    group_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
                else:
                    group_advantages[start_idx:end_idx] = group_rewards - group_mean
            
            advantages += group_advantages
        
        self.advantages = advantages
    
    def get_training_batch(self, step_idx: int) -> Dict[str, Any]:
        """
        获取指定时间步的训练数据批次。
        
        Args:
            step_idx: 时间步索引
            
        Returns:
            包含该时间步训练所需数据的字典
        """
        return {
            "latents": self.latents[:, step_idx] if self.latents is not None else None,
            "next_latents": self.next_latents[:, step_idx] if self.next_latents is not None else None,
            "log_probs": self.log_probs[:, step_idx] if self.log_probs is not None else None,
            "timesteps": self.timesteps[:, step_idx] if self.timesteps is not None else None,
            "advantages": self.advantages[:, step_idx] if self.advantages is not None else None,
            "encoder_hidden_states": self.encoder_hidden_states,
            "pooled_prompt_embeds": self.pooled_prompt_embeds,
            "text_ids": self.text_ids,
            "image_ids": self.image_ids,
        }
    
    def __repr__(self) -> str:
        info = [f"RolloutBuffer(batch_size={self.batch_size}, num_steps={self.num_steps}"]
        
        if self.latents is not None:
            info.append(f"  latents: {self.latents.shape}")
        if self.rewards is not None:
            info.append(f"  rewards: {len(self.rewards)} granularities")
        if self.advantages is not None:
            info.append(f"  advantages: {self.advantages.shape}")
        
        return "\n".join(info) + "\n)"

