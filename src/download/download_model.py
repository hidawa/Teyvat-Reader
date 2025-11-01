from huggingface_hub import snapshot_download

def download_deepseek_vl2(model_name: str = "deepseek-ai/deepseek-vl2-small",
                         local_dir: str = "./models/deepseek-vl2-small"):
    """
    Hugging Face から DeepSeek-VL2 モデルをローカルにダウンロードするスクリプト。
    """
    print(f"Downloading model {model_name} into {local_dir} …")
    snapshot_download(
        repo_id=model_name, 
        local_dir=local_dir,
    )
    print("Download completed.")

if __name__ == "__main__":
    # 例えば Tiny／Small／Full 版を選択可能
    download_deepseek_vl2("deepseek-ai/deepseek-vl2-small", "./models/deepseek-vl2-small")