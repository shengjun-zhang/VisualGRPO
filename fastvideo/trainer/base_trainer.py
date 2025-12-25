# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
BaseTrainer: 训练器的抽象基类。

定义了训练器需要实现的接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    训练器抽象基类。
    
    负责：
    1. 连接 DataLoader
    2. 调用 Sampler 进行 rollout
    3. 调用 RewardManager 打分
    4. 计算 Advantage
    5. 执行核心更新逻辑
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        transformer: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        device: torch.device,
    ):
        """
        初始化训练器。
        
        Args:
            config: 训练配置
            transformer: Diffusion transformer 模型
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            train_dataloader: 训练数据加载器
            device: 计算设备
        """
        self.config = config
        self.transformer = transformer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.device = device
        
        self.global_step = 0
        self.epoch = 0
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行单步训练。
        
        Args:
            batch: 数据批次
            
        Returns:
            包含损失和其他指标的字典
        """
        pass
    
    @abstractmethod
    def train(self, num_steps: int) -> None:
        """
        执行完整训练。
        
        Args:
            num_steps: 训练步数
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点。"""
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点。"""
        pass


