# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
BaseSampler: 采样器的抽象基类。

定义了采样器需要实现的接口，支持扩展到不同的 Diffusion 模型。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

from fastvideo.workers.rollout_buffer import RolloutBuffer


class BaseSampler(ABC):
    """
    采样器抽象基类。
    
    负责：
    1. 管理采样过程中产生的 latents、rewards 等数据
    2. 调用 Base Model 进行 rollout
    3. 返回结构化的 RolloutBuffer 对象
    """
    
    def __init__(
        self,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """
        初始化采样器。
        
        Args:
            transformer: Diffusion 模型的 transformer
            vae: VAE 模型，用于 latent <-> image 转换
            device: 计算设备
            config: 采样配置
        """
        self.transformer = transformer
        self.vae = vae
        self.device = device
        self.config = config
    
    @abstractmethod
    def sample(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        captions: list,
        reward_manager: Optional[Any] = None,
    ) -> RolloutBuffer:
        """
        执行采样，返回 RolloutBuffer。
        
        Args:
            encoder_hidden_states: 文本编码
            pooled_prompt_embeds: 池化的 prompt embeddings
            text_ids: 文本 IDs
            captions: 原始文本
            reward_manager: 奖励管理器（可选）
            
        Returns:
            包含采样结果的 RolloutBuffer
        """
        pass
    
    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> list:
        """
        将 latents 解码为图像。
        
        Args:
            latents: latent 表示
            
        Returns:
            PIL 图像列表
        """
        pass
    
    def set_eval_mode(self):
        """设置模型为评估模式。"""
        self.transformer.eval()
    
    def set_train_mode(self):
        """设置模型为训练模式。"""
        self.transformer.train()


