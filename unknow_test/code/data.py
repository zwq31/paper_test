from datasets import load_dataset

dataset = load_dataset("trivia_qa", "rc.wikipedia")

sample = dataset["validation"][0]
print(sample)

# dataset.save_to_disk("/mnt/project/unknow_test/data/triviaqa_rc_wikipedia")


# from modelscope import snapshot_download

# # 模型名写魔塔社区上的模型标识
# model_id = "shakechen/Llama-2-7b-chat-hf"

# # 下载到本地 cache_dir
# local_dir = snapshot_download(model_id, cache_dir='/mnt/project/unknow_test/model/Llama-2-7b-chat-hf')
# print("Model downloaded to:", local_dir)

