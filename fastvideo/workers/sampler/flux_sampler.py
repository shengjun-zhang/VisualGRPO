# Copyright (c) [2025] [VisualGRPO Team]
# SPDX-License-Identifier: Apache License 2.0

"""
FluxSampler: Flux 模型的采样器实现。

封装了 Flux 模型的采样逻辑，包括：
- Anchor 采样（ODE）
- SDE 采样（带噪声注入）
- ODE 采样（确定性）
"""

import math
import os
from typing import Dict, Any, Optional, List, Tuple
import torch
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from .base_sampler import BaseSampler
from fastvideo.workers.rollout_buffer import RolloutBuffer


def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """SD3 时间偏移函数。"""
    return (shift * t) / (1 + (shift - 1) * t)


def prepare_latent_image_ids(
    batch_size: int, 
    height: int, 
    width: int, 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    """准备 latent 图像 IDs。"""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.to(device=device, dtype=dtype)


def pack_latents(
    latents: torch.Tensor, 
    batch_size: int, 
    num_channels_latents: int, 
    height: int, 
    width: int
) -> torch.Tensor:
    """将 latents 打包为 Flux 格式。"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(
    latents: torch.Tensor, 
    height: int, 
    width: int, 
    vae_scale_factor: int
) -> torch.Tensor:
    """将 Flux 格式的 latents 解包。"""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


class FluxSampler(BaseSampler):
    """
    Flux 模型的采样器实现。
    
    支持：
    - 多步 SDE 采样（Merging Step）
    - 多粒度奖励计算
    - Anchor 采样用于评估
    """
    
    def __init__(
        self,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        super().__init__(transformer, vae, device, config)
        
        # 采样配置
        self.height = config.get("height", 720)
        self.width = config.get("width", 720)
        self.sampling_steps = config.get("sampling_steps", 16)
        self.shift = config.get("shift", 3.0)
        self.eta = config.get("eta", 0.7)
        self.guidance_scale = config.get("guidance_scale", 3.5)
        self.num_generations = config.get("num_generations", 12)
        self.init_same_noise = config.get("init_same_noise", True)
        
        # GRPO 配置
        self.eta_step_list = config.get("eta_step_list", [1])
        self.eta_step_merge_list = config.get("eta_step_merge_list", [1])
        self.granular_list = config.get("granular_list", [1, 2])
        
        # 常量
        self.SPATIAL_DOWNSAMPLE = 8
        self.IN_CHANNELS = 16
        self.VAE_SCALE_FACTOR = 8
        
        # 计算 latent 尺寸
        self.latent_h = self.height // self.SPATIAL_DOWNSAMPLE
        self.latent_w = self.width // self.SPATIAL_DOWNSAMPLE
        
        # 图像处理器
        self.vae.enable_tiling()
        self.image_processor = VaeImageProcessor(16)
    
    def _create_sigma_schedule(self) -> torch.Tensor:
        """创建 sigma 调度表。"""
        sigma_schedule = torch.linspace(1, 0, self.sampling_steps + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule)
        return sigma_schedule
    
    def _flow_grpo_step(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: Optional[torch.Tensor] = None,
        merge_step: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行单步 Flow GRPO 采样。
        
        Args:
            model_output: 模型输出
            latents: 当前 latents
            eta: 噪声系数
            sigmas: sigma 调度表
            index: 当前步索引
            prev_sample: 预设的下一步样本（用于计算 log_prob）
            merge_step: 合并步数
            generator: 随机数生成器
            
        Returns:
            prev_sample: 下一步样本
            pred_original_sample: 预测的原始样本
            log_prob: 对数概率
        """
        device = model_output.device
        sigma = sigmas[index].to(device)
        sigma_prev = sigmas[index + merge_step].to(device)
        sigma_max = sigmas[1].item()
        dt = sigma_prev - sigma  # 负 dt

        pred_original_sample = latents - sigma * model_output
        
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
        
        prev_sample_mean = (
            latents * (1 + std_dev_t**2 / (2 * sigma) * dt) + 
            model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape, 
                generator=generator, 
                device=device, 
                dtype=model_output.dtype
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
        
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, pred_original_sample, log_prob
    
    def _get_transformer_output(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        timesteps: torch.Tensor,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """获取 transformer 输出。"""
        # 关键：始终使用 train 模式，FSDP 在 eval 模式下可能有同步问题
        self.transformer.train()
        
        with torch.autocast("cuda", torch.bfloat16):
            if train_mode:
                pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor(
                        [self.guidance_scale],
                        device=latents.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids if image_ids.dim() == 2 else image_ids.squeeze(0),
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                with torch.no_grad():
                    pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timesteps / 1000,
                        guidance=torch.tensor(
                            [self.guidance_scale],
                            device=latents.device,
                            dtype=torch.bfloat16
                        ),
                        txt_ids=text_ids,
                        pooled_projections=pooled_prompt_embeds,
                        img_ids=image_ids if image_ids.dim() == 2 else image_ids.squeeze(0),
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
        
        return pred
    
    def _run_anchor_sample(
        self,
        z: torch.Tensor,
        sigma_schedule: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        运行 anchor 采样（确定性 ODE 采样）。
        """
        import sys
        all_latents = [z]
        rank = int(os.environ.get("RANK", 0))
        
        if rank == 0:
            print(f"[DEBUG] _run_anchor_sample entered")
            print(f"[DEBUG] z: {z.shape}, {z.device}")
            print(f"[DEBUG] encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"[DEBUG] text_ids: {text_ids.shape}")
            print(f"[DEBUG] image_ids: {image_ids.shape}")
            sys.stdout.flush()
        
        for i in range(self.sampling_steps):
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full(
                [encoder_hidden_states.shape[0]], 
                timestep_value, 
                device=z.device, 
                dtype=torch.long
            )
            
            if rank == 0 and i == 0:
                print(f"[DEBUG] Step {i}: Before transformer call")
                sys.stdout.flush()
            
            # 使用 train 模式
            self.transformer.train()
            with torch.autocast("cuda", torch.bfloat16):
                pred = self.transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor(
                        [self.guidance_scale],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            
            if rank == 0 and i == 0:
                print(f"[DEBUG] Step {i}: After transformer call, pred: {pred.shape}")
                sys.stdout.flush()
            
            z, pred_original, _ = self._flow_grpo_step(
                pred, z.to(torch.float32), 0, 
                sigmas=sigma_schedule, index=i, prev_sample=None
            )
            all_latents.append(z.to(torch.bfloat16))
            
            if rank == 0:
                print(f"[DEBUG] Completed step {i}/{self.sampling_steps}")
                sys.stdout.flush()
        
        all_latents = torch.stack(all_latents, dim=1)
        return pred_original, all_latents
    
    def _run_sde_sample_step(
        self,
        z: torch.Tensor,
        eta_step: int,
        sigma_schedule: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        merge_step: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        运行单步 SDE 采样（带噪声注入）。
        
        Returns:
            all_latents: 形状 [B, 2, ...]，包含输入和输出 latent
            all_log_probs: 形状 [B, 1]，对数概率
        """
        all_latents = [z]
        all_log_probs = []
        
        sigma = sigma_schedule[eta_step]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full(
            [encoder_hidden_states.shape[0]], 
            timestep_value, 
            device=z.device, 
            dtype=torch.long
        )
        
        with torch.no_grad():
            pred = self._get_transformer_output(
                z, encoder_hidden_states, pooled_prompt_embeds,
                text_ids, image_ids, timesteps, train_mode=False
            )
        
        z, pred_original, log_prob = self._flow_grpo_step(
            pred, z.to(torch.float32), self.eta,
            sigmas=sigma_schedule, index=eta_step, 
            prev_sample=None, merge_step=merge_step
        )
        
        all_latents.append(z.to(torch.bfloat16))
        all_log_probs.append(log_prob)
        
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        
        return all_latents, all_log_probs
    
    def _run_ode_sample(
        self,
        z: torch.Tensor,
        start_step: int,
        sigma_schedule: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        运行 ODE 采样（确定性采样，用于最终图像生成）。
        
        Returns:
            pred_original: 最终预测的原始样本
        """
        sample_steps = len(sigma_schedule)
        progress_bar = tqdm(range(start_step + 1, sample_steps - 1), desc="ODE Sampling")
        
        for i in progress_bar:
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full(
                [encoder_hidden_states.shape[0]], 
                timestep_value, 
                device=z.device, 
                dtype=torch.long
            )
            
            with torch.no_grad():
                pred = self._get_transformer_output(
                    z, encoder_hidden_states, pooled_prompt_embeds,
                    text_ids, image_ids, timesteps, train_mode=False
                )
            
            z, pred_original, _ = self._flow_grpo_step(
                pred, z.to(torch.float32), 0,
                sigmas=sigma_schedule, index=i, prev_sample=None
            )
        
        return pred_original
    
    def decode_latents(self, latents: torch.Tensor) -> List:
        """将 latents 解码为图像。"""
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, self.height, self.width, self.VAE_SCALE_FACTOR)
                latents = (latents / 0.3611) + 0.1159
                image = self.vae.decode(latents, return_dict=False)[0]
                decoded_images = self.image_processor.postprocess(image)
        return decoded_images
    
    def sample(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        captions: list,
        reward_manager: Optional[Any] = None,
    ) -> RolloutBuffer:
        """
        执行完整的采样流程。
        
        Args:
            encoder_hidden_states: 文本编码，形状 [B, L, D]
            pooled_prompt_embeds: 池化的 prompt embeddings
            text_ids: 文本 IDs
            captions: 原始文本
            reward_manager: 奖励管理器（可选）
            
        Returns:
            包含采样结果的 RolloutBuffer
        """
        # 确保数据在正确的设备上
        encoder_hidden_states = encoder_hidden_states.to(self.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)
        
        # 注意：预处理脚本中 text_ids 的保存有问题，这里直接根据 encoder_hidden_states 创建正确的 text_ids
        # text_ids 应该是 [seq_len, 3] 的形状，表示文本序列的位置编码（全零即可）
        text_seq_len = encoder_hidden_states.shape[1]  # 文本序列长度，通常是 512
        text_ids = torch.zeros(text_seq_len, 3, device=self.device, dtype=torch.bfloat16)
        
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"[DEBUG] sample() - input shapes:")
            print(f"[DEBUG]   encoder_hidden_states: {encoder_hidden_states.shape}")
            print(f"[DEBUG]   pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
            print(f"[DEBUG]   text_ids (recreated): {text_ids.shape}")
        
        sigma_schedule = self._create_sigma_schedule().to(self.device)
        B = encoder_hidden_states.shape[0]
        batch_size = 1  # 每次处理一个样本
        batch_indices = torch.chunk(torch.arange(B), B // batch_size)
        granular_nums = len(self.granular_list)
        
        # 初始化收集器
        all_input_latents = []
        all_output_latents = []
        all_log_probs = []
        all_rewards = [[] for _ in range(granular_nums)]
        all_image_ids = []
        eval_rewards = []
        
        # 初始化噪声（可选择同一噪声）
        if self.init_same_noise:
            input_latents = torch.randn(
                (1, self.IN_CHANNELS, self.latent_h, self.latent_w),
                device=self.device,
                dtype=torch.bfloat16,
            )
        
        anchor_latents = None
        
        for index, batch_idx in enumerate(batch_indices):
            batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
            batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
            batch_captions = [captions[i] for i in batch_idx]
            
            image_ids = prepare_latent_image_ids(
                len(batch_idx), self.latent_h // 2, self.latent_w // 2,
                self.device, torch.bfloat16
            )
            
            # text_ids 已经在上面创建为正确的 [seq_len, 3] 格式，直接使用
            txt_ids_for_transformer = text_ids
            
            if rank == 0 and index == 0:
                print(f"[DEBUG] txt_ids_for_transformer shape: {txt_ids_for_transformer.shape}")
                print(f"[DEBUG] image_ids shape: {image_ids.shape}")
                import sys; sys.stdout.flush()
            
            step_rewards = [[] for _ in range(granular_nums)]
            step_input_latents = []
            step_output_latents = []
            step_log_probs = []
            
            # 每 num_generations 个样本进行一次 anchor 采样
            if index % self.num_generations == 0:
                if rank == 0:
                    print(f"[DEBUG] Before pack_latents, index={index}")
                    import sys; sys.stdout.flush()
                
                input_latents_packed = pack_latents(
                    input_latents, len(batch_idx), self.IN_CHANNELS,
                    self.latent_h, self.latent_w
                )
                
                if rank == 0:
                    print(f"[DEBUG] After pack_latents, shape={input_latents_packed.shape}")
                    print(f"[DEBUG] Calling _run_anchor_sample...")
                    import sys; sys.stdout.flush()
                
                with torch.no_grad():
                    eval_latents, anchor_latents = self._run_anchor_sample(
                        input_latents_packed, sigma_schedule,
                        batch_encoder_hidden_states, batch_pooled_prompt_embeds,
                        txt_ids_for_transformer, image_ids
                    )
                    
                    # 计算 anchor 的奖励用于评估
                    if reward_manager is not None:
                        eval_images = self.decode_latents(eval_latents)
                        eval_reward = reward_manager.compute_rewards(
                            eval_images, batch_captions
                        )
                        eval_rewards.append(torch.tensor(eval_reward, device=self.device))
                
                # 对每个 eta_step 进行 SDE 采样
                for i, eta_step in enumerate(self.eta_step_list):
                    input_sde_sample = anchor_latents[:, eta_step]
                    merge_step = self.eta_step_merge_list[i]
                    
                    batch_latents, batch_log_probs = self._run_sde_sample_step(
                        input_sde_sample, eta_step, sigma_schedule,
                        batch_encoder_hidden_states, batch_pooled_prompt_embeds,
                        txt_ids_for_transformer, image_ids, merge_step=merge_step
                    )
                    
                    input_ode_latents = batch_latents[:, 1]
                    
                    # 对每个粒度计算奖励
                    for j, g in enumerate(self.granular_list):
                        prefix = sigma_schedule[:eta_step + merge_step + 1]
                        suffix = sigma_schedule[eta_step + merge_step + 1::g]
                        sigma_schedule_j = torch.cat((prefix, suffix))
                        
                        latents_j = self._run_ode_sample(
                            input_ode_latents, eta_step,
                            sigma_schedule_j,
                            batch_encoder_hidden_states, batch_pooled_prompt_embeds,
                            txt_ids_for_transformer, image_ids
                        )
                        
                        # 计算奖励
                        if reward_manager is not None:
                            images_j = self.decode_latents(latents_j)
                            rewards_j = reward_manager.compute_rewards(
                                images_j, batch_captions
                            )
                            step_rewards[j].append(
                                torch.tensor(rewards_j, device=self.device)
                            )
                    
                    step_input_latents.append(batch_latents[:, 0])
                    step_output_latents.append(batch_latents[:, 1])
                    step_log_probs.append(batch_log_probs[:, 0])
            
            # 聚合当前批次的结果
            for j in range(granular_nums):
                if step_rewards[j]:
                    step_rewards[j] = torch.stack(step_rewards[j], dim=1)
                    all_rewards[j].append(step_rewards[j])
            
            all_input_latents.append(torch.stack(step_input_latents, dim=1))
            all_output_latents.append(torch.stack(step_output_latents, dim=1))
            all_log_probs.append(torch.stack(step_log_probs, dim=1))
            all_image_ids.append(image_ids)
        
        # 构建 RolloutBuffer
        rollout_buffer = RolloutBuffer(
            latents=torch.cat(all_input_latents, dim=0),
            next_latents=torch.cat(all_output_latents, dim=0),
            log_probs=torch.cat(all_log_probs, dim=0),
            rewards=[torch.cat(r, dim=0).to(torch.float32) for r in all_rewards if r],
            eval_rewards=torch.cat(eval_rewards, dim=0).to(torch.float32) if eval_rewards else None,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=torch.stack(all_image_ids, dim=0),
            captions=captions,
            sigma_schedule=sigma_schedule,
            device=self.device,
            batch_size=B,
            num_steps=len(self.eta_step_list),
        )
        
        # 构建 timesteps
        train_sigma_schedule = sigma_schedule.clone()[self.eta_step_list]
        timestep_values = [int(sigma * 1000) for sigma in train_sigma_schedule]
        timesteps = torch.tensor(
            [timestep_values for _ in range(B)],
            device=self.device, dtype=torch.long
        )
        rollout_buffer.timesteps = timesteps
        
        return rollout_buffer
    
    def get_pred_for_training(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取用于训练的模型预测。
        
        与采样时不同，训练时需要开启梯度计算。
        """
        return self._get_transformer_output(
            latents, encoder_hidden_states, pooled_prompt_embeds,
            text_ids, image_ids, timesteps, train_mode=True
        )
    
    def compute_log_prob(
        self,
        pred: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        sigma_schedule: torch.Tensor,
        eta_step: int,
        merge_step: int,
    ) -> torch.Tensor:
        """计算给定预测的对数概率。"""
        _, _, log_prob = self._flow_grpo_step(
            pred, latents.float(), self.eta,
            sigma_schedule, eta_step,
            prev_sample=next_latents.float(),
            merge_step=merge_step
        )
        return log_prob


