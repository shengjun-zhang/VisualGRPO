# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
RewardManager: 奖励管理器。

负责加载和调用各种奖励模型，计算生成图像的奖励分数。
支持：
- HPS (Human Preference Score)
- CLIP Score
- 多种奖励模型的组合
"""

from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch


class RewardManager:
    """
    奖励管理器，封装奖励模型的加载和调用。
    
    支持单一奖励模型或多种奖励模型的组合。
    """
    
    def __init__(
        self,
        device: torch.device,
        reward_config: Dict[str, Any],
    ):
        """
        初始化奖励管理器。
        
        Args:
            device: 计算设备
            reward_config: 奖励模型配置，包含：
                - type: 奖励模型类型 ("hps", "clip", "combined")
                - model_path: 模型路径
                - 其他模型特定配置
        """
        self.device = device
        self.reward_config = reward_config
        self.reward_type = reward_config.get("type", "hps")
        
        # 初始化奖励模型
        self.reward_models = {}
        self._init_reward_models()
    
    def _init_reward_models(self):
        """根据配置初始化奖励模型。"""
        if self.reward_type == "hps":
            self._init_hps_model()
        elif self.reward_type == "clip":
            self._init_clip_model()
        elif self.reward_type == "combined":
            # 支持多种奖励模型的组合
            for model_config in self.reward_config.get("models", []):
                model_type = model_config.get("type")
                if model_type == "hps":
                    self._init_hps_model(model_config)
                elif model_type == "clip":
                    self._init_clip_model(model_config)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _init_hps_model(self, config: Optional[Dict] = None):
        """初始化 HPS 奖励模型。"""
        config = config or self.reward_config
        
        # 尝试不同的导入路径
        try:
            # 优先使用本地克隆的 HPSv2
            from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError:
            try:
                # 尝试 pip 安装的 hpsv2
                from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            except ImportError:
                raise ImportError(
                    "HPSv2 not found. Please install it:\n"
                    "  1. Clone: git clone https://github.com/tgxs002/HPSv2.git\n"
                    "  2. Or: pip install hpsv2"
                )
        
        model_name = config.get("model_name", "ViT-H-14")
        clip_path = config.get("clip_path")
        hps_path = config.get("hps_path")
        
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            clip_path,
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        
        # 加载 HPS 权重
        if isinstance(self.device, int):
            map_location = f'cuda:{self.device}'
        else:
            map_location = self.device
        
        checkpoint = torch.load(hps_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        
        tokenizer = get_tokenizer(model_name)
        model = model.to(self.device)
        model.eval()
        
        self.reward_models["hps"] = {
            "model": model,
            "tokenizer": tokenizer,
            "preprocess": preprocess_val,
            "weight": config.get("weight", 1.0),
        }
    
    def _init_clip_model(self, config: Optional[Dict] = None):
        """初始化 CLIP 奖励模型。"""
        config = config or self.reward_config
        
        import open_clip
        
        model_name = config.get("model_name", "ViT-H-14")
        pretrained = config.get("pretrained", "laion2b_s32b_b79k")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        model = model.to(self.device)
        model.eval()
        
        self.reward_models["clip"] = {
            "model": model,
            "tokenizer": tokenizer,
            "preprocess": preprocess,
            "weight": config.get("weight", 1.0),
        }
    
    @torch.no_grad()
    def compute_rewards(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ) -> List[float]:
        """
        计算图像的奖励分数。
        
        Args:
            images: PIL 图像或图像列表
            texts: 对应的文本描述
            
        Returns:
            奖励分数列表
        """
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        # 收集所有奖励模型的分数
        all_rewards = []
        
        for model_name, model_info in self.reward_models.items():
            if model_name == "hps":
                rewards = self._compute_hps_rewards(images, texts, model_info)
            elif model_name == "clip":
                rewards = self._compute_clip_rewards(images, texts, model_info)
            else:
                continue
            
            weight = model_info.get("weight", 1.0)
            all_rewards.append([r * weight for r in rewards])
        
        # 合并多个奖励模型的分数
        if not all_rewards:
            return [0.0] * len(images)
        
        combined_rewards = []
        for i in range(len(images)):
            combined_rewards.append(sum(r[i] for r in all_rewards))
        
        return combined_rewards
    
    def _compute_hps_rewards(
        self,
        images: List[Image.Image],
        texts: List[str],
        model_info: Dict,
    ) -> List[float]:
        """计算 HPS 奖励分数。"""
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        preprocess = model_info["preprocess"]
        
        rewards = []
        for image, text in zip(images, texts):
            image_tensor = preprocess(image).unsqueeze(0).to(
                device=self.device, non_blocking=True
            )
            text_tensor = tokenizer([text]).to(device=self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(image_tensor, text_tensor)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)
                rewards.append(hps_score.float().cpu().item())
        
        return rewards
    
    def _compute_clip_rewards(
        self,
        images: List[Image.Image],
        texts: List[str],
        model_info: Dict,
    ) -> List[float]:
        """计算 CLIP 奖励分数。"""
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        preprocess = model_info["preprocess"]
        
        rewards = []
        for image, text in zip(images, texts):
            image_tensor = preprocess(image).unsqueeze(0).to(
                device=self.device, non_blocking=True
            )
            text_tensor = tokenizer([text]).to(device=self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tensor)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).squeeze()
                rewards.append(similarity.float().cpu().item())
        
        return rewards
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取已加载的奖励模型信息。"""
        info = {
            "type": self.reward_type,
            "models": list(self.reward_models.keys()),
        }
        return info


