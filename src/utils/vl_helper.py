from huggingface_hub import InferenceClient

# DeepSeek-VLモデルをHF経由で呼び出す例
client = InferenceClient("deepseek-ai/DeepSeek-VL-7B")

async def analyze_with_deepseek_vl(file):
    """DeepSeek-VLを呼び出して画像内容を理解させる"""
    with open(file.filename, "rb") as f:
        image_bytes = f.read()
    prompt = "この画像の人物の特徴・ポーズ・文化的モチーフを日本語で説明してください。"
    result = client.text_to_image(prompt=prompt, image=image_bytes)
    return result
