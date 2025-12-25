from typing import Union, List
import argparse
import torch
from PIL import Image

from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPSClipRewardModel(object):
    def __init__(self, device, clip_ckpt_path, hps_ckpt_path, model_name='ViT-H-14'):
        self.device = device
        self.clip_ckpt_path = clip_ckpt_path
        self.hps_ckpt_path = hps_ckpt_path
        self.model_name = model_name
        self.reward_model, self.text_processor, self.img_processor = self.build_reward_model()

    def build_reward_model(self):
        model, preprocess_train, img_preprocess_val = create_model_and_transforms(
            self.model_name,
            self.clip_ckpt_path,
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

        # Convert device name to proper format
        if isinstance(self.device, int):
            ml_device = str(self.device)
        else:
            ml_device = self.device

        if not ml_device.startswith('cuda'):
            ml_device = f'cuda:{ml_device}' if ml_device.isdigit() else ml_device

        checkpoint = torch.load(self.hps_ckpt_path, map_location=ml_device)
        model.load_state_dict(checkpoint['state_dict'])
        text_processor = get_tokenizer(self.model_name)
        reward_model = model.to(self.device)
        reward_model.eval()

        return reward_model, text_processor, img_preprocess_val

    @torch.no_grad()
    def __call__(
            self,
            images: Union[Image.Image, List[Image.Image]],
            texts: Union[str, List[str]],
    ):
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]
        
        rewards = []
        for image, text in zip(images, texts):
            image = self.img_processor(image).unsqueeze(0).to(self.device, non_blocking=True)
            text = self.text_processor([text]).to(device=self.device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = self.reward_model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)
                
                # reward is a tensor of shape (1,) --> list
                rewards.append(hps_score.float().cpu().item())
        
        return rewards
