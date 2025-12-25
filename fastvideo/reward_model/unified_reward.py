import argparse
import base64
import os
import re
import requests
import time
import concurrent.futures
from io import BytesIO
from typing import List, Optional, Union

from PIL import Image


QUESTION_TEMPLATE_SEMANTIC = (
    "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
    "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
    "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
    "   - A higher score indicates that the word is less well represented in the image.\n"
    "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
    "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "---\n\n"
    "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\nText Caption: [{}]"
)

QUESTION_TEMPLATE_SCORE = (
    "You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n"
    "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
    "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\n"
    "Extract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\n"
    "Your task is provided as follows:\nText Caption: [{}]"
)


class VLMessageClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self):
        """Close the session if it exists."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _encode_image_base64(self, image):
        if isinstance(image, str):
            with Image.open(image) as img:
                img = img.convert("RGB")
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
    def build_messages(self, item, image_root=""):
        if isinstance(item['image'], str):
            image_path = os.path.join(image_root, item['image'])
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                        {
                            "type": "text",
                            "text": f"{item['question']}"
                        }
                    ]
                }
            ]
        assert isinstance(item['image'], Image.Image), f"image must be a PIL.Image.Image, but got {type(item['image'])}"
        return [    
            {
                "role": "user",
                "content": [
                    {"type": "pil_image", "pil_image": item['image']},
                    {
                        "type": "text",
                        "text": f"{item['question']}"
                    }
                ]
            }
        ]

    def format_messages(self, messages):
        formatted = []
        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}

            if msg["role"] == "system":
                new_msg["content"] = msg["content"][0]["text"]
            else:
                for part in msg["content"]:
                    if part["type"] == "image_url":
                        img_path = part["image_url"]["url"].replace("file://", "")
                        base64_image = self._encode_image_base64(img_path)
                        new_part = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                        new_msg["content"].append(new_part)
                    elif part["type"] == "pil_image":
                        base64_image = self._encode_image_base64(part["pil_image"])
                        new_part = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                        new_msg["content"].append(new_part)
                    else:
                        new_msg["content"].append(part)
            formatted.append(new_msg)
        return formatted

    def process_item(self, item, image_root=""):
        max_retries = 3
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1
                raw_messages = self.build_messages(item, image_root)
                formatted_messages = self.format_messages(raw_messages)

                payload = {
                    "model": "UnifiedReward",
                    "messages": formatted_messages,
                    "temperature": 0,
                    "max_tokens": 4096,
                }

                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=30 + attempt*5 
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]

                result = {
                    "question": item["question"],
                    "image_path": item["image"] if isinstance(item["image"], str) else "PIL_Image",
                    "model_output": output,
                    "attempt": attempt,
                    "success": True
                }
                break  

            except Exception as e:
                if attempt == max_retries:
                    result = {
                        "question": item["question"],
                        "image_path": item["image"] if isinstance(item["image"], str) else "PIL_Image",
                        "error": str(e),
                        "attempt": attempt,
                        "success": False
                    }
                    raise(e)
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)

        return result, result.get("success", False)


class UnifiedRewardModel(object):
    def __init__(self, api_url, default_question_type="score", num_workers=8):
        self.api_url = api_url
        self.num_workers = num_workers
        self.default_question_type = default_question_type
        self.question_template_score = QUESTION_TEMPLATE_SCORE
        self.question_template_semantic = QUESTION_TEMPLATE_SEMANTIC
        # self.client = VLMessageClient(self.api_url)
    
    def question_constructor(self, prompt, question_type=None):
        if question_type is None:
            question_type = self.default_question_type
        if question_type == "score":
            return self.question_template_score.format(prompt)
        elif question_type == "semantic":
            return self.question_template_semantic.format(prompt)
        else:
            raise ValueError(f"Invalid question type: {question_type}")

    def _process_item_wrapper(self, client, image, question):
        try:
            item = {
                "image": image,
                "question": question,
            }
            result, _ = client.process_item(item)
            return result
        except Exception as e:
            print(f"Encountered error in unified reward model processing: {str(e)}")
            return None
    
    def _reset_proxy(self):
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)

    def __call__(self, 
            images: Union[List[Image.Image], List[str]],
            prompts: Union[str, List[str]],
            question_type: Optional[str] = None,
    ):
        # Reset proxy, otherwise cannot access the server url
        self._reset_proxy()
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("prompts must have the same length as images")
        
        with VLMessageClient(self.api_url) as client:
            questions = [self.question_constructor(prompt, question_type) for prompt in prompts]
            
            # Initialize results and successes lists with None and False
            results = [None] * len(images)
            successes = [False] * len(images)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks and keep track of their order
                future_to_idx = {
                    executor.submit(self._process_item_wrapper, client, image, question): idx 
                    for idx, (image, question) in enumerate(zip(images, questions))
                }
                
                # Get results in completion order but store them in the correct position
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    if result is not None and result.get("success", False):
                        output = result.get("model_output", "")
                        score = self.score_parser(output, question_type)
                        results[idx] = score
                        successes[idx] = True
                    else:
                        results[idx] = None
                        successes[idx] = False

            return results, successes
    
    def score_parser(self, text, question_type=None):
        if question_type is None:
            question_type = self.default_question_type
        if question_type == "score":
            return self.extract_final_score(text)
        elif question_type == "semantic":
            return self.extract_alignment_score(text)
        else:
            raise ValueError(f"Invalid question type: {question_type}")
    
    @staticmethod
    def extract_alignment_score(text):
        """
        Extract Alignment Score (1-5) from the evaluation text.
        Returns a float score if found, None otherwise.
        """
        match = re.search(r'Alignment Score \(1-5\):\s*([0-5](?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        else:
            return None

    @staticmethod
    def extract_final_score(text):
        """
        Extract Final Score from the evaluation text.
        Returns a float score if found, None otherwise.
        Example input:
            'ocean (location): 0
            clouds (object): 1
            birds (animal): 0
            day time (attribute): 1
            low depth field effect (attribute): 1
            painting (attribute): 1
            Final Score: 2.33'
        """
        match = re.search(r'Final Score:\s*([0-5](?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        else:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--max_workers", type=int)
    args = parser.parse_args()

    unified_reward_model = UnifiedRewardModel(args.api_url, num_workers=args.max_workers)
    img_path = "assets/reward_demo.jpg"
    images = [
        Image.open(img_path).convert("RGB")
        for i in range(1, 5)
    ] * 4
    prompts = "A 3D rendering of anime schoolgirls with a sad expression underwater, surrounded by dramatic lighting."
    results, successes = unified_reward_model(images, prompts, question_type="semantic")
    print(results)
    print(successes)

    # # 并发测试
    # proc_num = 32
    
    # for i in range(5):
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=proc_num) as executor:
    #         futures = [executor.submit(unified_reward_model, images, prompts, question_type="semantic") for _ in range(proc_num)]
    #         results = [future.result() for future in concurrent.futures.as_completed(futures)]
    #     print(results)