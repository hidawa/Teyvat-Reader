from typing import Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from PIL import Image

# repo root から相対パスで指定
ROOT = Path(__file__).parents[3]
DEFAULT_MODEL_DIR = (ROOT / "models" / "deepseek-vl2-tiny").resolve()

class DeepSeekVL2Local:
    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        print("ising device is ", self.device)

    def load(self, dtype=torch.float16):
        if self.model is not None:
            return

        print(f"[DeepSeekVL2Local] loading model from {self.model_dir}")
        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(str(self.model_dir))
        self.tokenizer = self.processor.tokenizer

        torch_dtype = dtype if self.device.startswith("cuda") else torch.float16
        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ) # .to(self.device) device_map="auto"では不要
        self.model.eval()
    
    def analyze_image(self, image_path: Optional[str], prompt: str, max_new_tokens: int = 256):
        """
        Analyzes an image with given prompt using a pre-trained vision-language model.

        Args:
            image_path (Optional[str]): Path to the image file to analyze. If None, performs text-only analysis.
            prompt (str): Text prompt to guide the analysis of the image.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.

        Returns:
            str: Generated text response from the model.

        Notes:
            - self.model must be an instance of AutoModelForCausalLM
            - The image is converted to RGB format before processing
            - The model uses a conversation format with image and text inputs
            - Generation is performed without gradient calculation
        """
        if self.model is None:
            self.load()

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        else:
            # ダミー画像（真っ白 1x1）
            image = Image.new("RGB", (1, 1), color=(255, 255, 255))
        # プロンプトと画像をモデル入力形式に変換
        conversation = [
            {
                "role": "<|User|>", 
                "content": f"{prompt.strip()}<|end|>",
                "images": [image]
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # text_prompt = self.processor.format_messages_v2(
        #     conversation,
        #     pil_images=[image],
        # )

        prepare_inputs = self.processor(
            # text=[text_prompt], 
            conversations=conversation,
            images=[image], 
            return_tensors="pt",
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         inputs_embeds=inputs_embeds, 
        #         max_new_tokens=max_new_tokens,
        #         attention_mask=prepare_inputs.attention_mask,
        #         pad_token_id=self.tokenizer.eos_token_id,
        #         bos_token_id=self.tokenizer.bos_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         do_sample=False,
        #         use_cache=True
        #     )
            # answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
            # response_text = answer.replace("<|endoftext|>", "").strip()  # EOSトークンを除去
        outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        response_text = answer.replace("<|endoftext|>", "").strip()  # EOSトークンを除去

        return response_text
