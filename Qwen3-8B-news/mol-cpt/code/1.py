from modelscope.hub.snapshot_download import snapshot_download

model_id = 'Qwen/Qwen3-8B'  # 确认模型ID正确
local_dir = '/mnt/project/Qwen3-8B-news/models/Qwen3-8B'

model_dir = snapshot_download(
    model_id,
    local_dir=local_dir,      # 仍然指定保存路径
    # 去掉 local_dir_use_symlinks 参数
)

print(f"模型已下载到: {model_dir}")