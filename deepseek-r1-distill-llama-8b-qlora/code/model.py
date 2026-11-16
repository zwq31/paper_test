from modelscope.hub.snapshot_download import snapshot_download

# 指定模型 ID（在 ModelScope 模型详情页可以找到）
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 替换为实际的模型 ID

# 指定下载路径
download_path = "~/project/deepseek-r1-distill-8b-qlora/model/"  # 替换为你希望保存模型的路径

# 下载模型到指定路径
model_dir = snapshot_download(model_id, cache_dir=download_path)

