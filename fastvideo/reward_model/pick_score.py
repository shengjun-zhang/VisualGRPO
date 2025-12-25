import os
import torch
import argparse
from typing import List, Tuple, Union
from transformers import AutoProcessor, AutoModel
from PIL import Image


class PickScoreRewardModel(object):
    def __init__(self, device: str = "cuda", http_proxy=None, https_proxy=None, mean=18.0, std=8.0):
        """
        Initialize PickScore reward model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.device = device
        self.processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        self.mean = mean
        self.std = std
        
        # Initialize model and processor
        self.processor = AutoProcessor.from_pretrained(self.processor_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_pretrained_name_or_path).eval().to(device)

    @torch.no_grad()
    def __call__(
            self,
            images: List[Image.Image],
            prompts: Union[str, List[str]],
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate probabilities and scores for images given a prompt.
        
        Args:
            prompts: Text prompt to evaluate images against
            images: List of PIL Images to evaluate
            
        Returns:
            Tuple of (probabilities, scores) for each image
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        scores = []
        for prompt, image in zip(prompts, images):
            # Preprocess images
            image_inputs = self.processor(
                images=[image],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            
            # Preprocess text
            text_inputs = self.processor(
                text=prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            # Get embeddings
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # Calculate scores
            score = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            score = (score - self.mean) / self.std
            scores.extend(score.cpu().tolist())
        
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PickScore Reward Model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--http_proxy", type=str, default=None, help="HTTP proxy URL")
    parser.add_argument("--https_proxy", type=str, default=None, help="HTTPS proxy URL")
    args = parser.parse_args()
    
    # Example usage
    reward_model = PickScoreRewardModel(
        device=args.device,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
    )
    pil_images = [Image.open("assets/reward_demo.jpg")]
    
    prompt = "A 3D rendering of anime schoolgirls with a sad expression underwater, surrounded by dramatic lighting."
    
    scores = reward_model(pil_images, [prompt] * len(pil_images))
    scores = [(s * reward_model.std + reward_model.mean) / 100.0 for s in scores]
    print(f"scores: {scores}")

