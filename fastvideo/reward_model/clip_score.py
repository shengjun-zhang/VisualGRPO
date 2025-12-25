import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import clip
from PIL import Image
from typing import List, Tuple, Union
from PIL import Image
import os
from open_clip import create_model_from_pretrained, get_tokenizer
import argparse 



@torch.no_grad()
def calculate_clip_score(prompts, images, clip_model, device):
    texts = clip.tokenize(prompts, truncate=True).to(device=device)

    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features)
    return scores


class CLIPScoreRewardModel():
    def __init__(self, clip_model_path, device, http_proxy=None, https_proxy=None, clip_model_type='ViT-H-14'):
        super().__init__()
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.clip_model_path = clip_model_path
        self.clip_model_type = clip_model_type
        self.device = device
        self.load_model()

    def load_model(self, logger=None):
        self.model, self.preprocess = create_model_from_pretrained(self.clip_model_path)
        self.tokenizer = get_tokenizer(self.clip_model_type)
        self.model.to(self.device)

    # calculate clip score directly, such as for rerank
    @torch.no_grad()
    def __call__(
        self, 
        prompts: Union[str, List[str]], 
        images: List[Image.Image]
    ) -> List[float]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        scores = []
        for prompt, image in zip(prompts, images):
            image_proc = self.preprocess(image).unsqueeze(0).to(self.device)
            text = self.tokenizer(
                [prompt], 
                context_length=self.model.context_length
            ).to(self.device)

            image_features = self.model.encode_image(image_proc)
            text_features = self.model.encode_text(text)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            clip_score = image_features @ text_features.T

            scores.append(clip_score.item())

        return scores
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PickScore Reward Model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--http_proxy", type=str, default=None, help="HTTP proxy URL")
    parser.add_argument("--https_proxy", type=str, default=None, help="HTTPS proxy URL")
    args = parser.parse_args()

    # Example usage
    clip_model_path = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
    reward_model = CLIPScoreRewardModel(
        clip_model_path, 
        device=args.device,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy
    )
    
    image_path = "assets/reward_demo.jpg"
    prompt = "A 3D rendering of anime schoolgirls with a sad expression underwater, surrounded by dramatic lighting."
    
    image = Image.open(image_path).convert("RGB")
    clip_score = reward_model(prompt, [image])

    print(f"CLIP Score: {clip_score}")