# from datasets import load_dataset
# import json

# # === 1. 加载数据集 ===
# print("正在加载 Magpie 数据集...")
# dataset = load_dataset("justus27/stage-1-lightr1-sft", split="train")
# dataset.save_to_disk("./mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/stage-1-lightr1-sft")
# print(f"✅ 已保存数据集到./mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/magpie_general")


# # === 3. 转换为 MoL 格式并保存 ===
# output_path = "../data/magpie_general_zh_mol.jsonl"
# count = 0

# with open(output_path, 'w', encoding='utf-8') as fout:
#     for sample in dataset:
#         instruction = sample.get("instruction", "").strip()
#         response = sample.get("response", "").strip()

#         if not instruction or not response:
#             continue  # 忽略空样本

#         item = {
#             "instruction": instruction,
#             "output": response,
#             "is_domain_batch": False
#         }

#         fout.write(json.dumps(item, ensure_ascii=False) + "\n")
#         count += 1

# print(f"✅ 已保存 {count} 条通用样本到 {output_path}")
# import json
# from pathlib import Path

# # 输入输出路径
# input_path = Path("../data/magpie_general_mol.jsonl")
# output_path = Path("../data/magpie_general_mol_messages.jsonl")

# def convert_instruction_to_messages(item):
#     """将 instruction + output 转成 messages 格式"""
#     user_msg = item.get("instruction", "").strip()
#     assistant_msg = item.get("output", "").strip()
#     return {
#         "messages": [
#             {"role": "system", "content":  "你是一个乐于助人的助手。You are a helpful assistant."},
#             {"role": "user", "content": user_msg},
#             {"role": "assistant", "content": assistant_msg}
#         ],
#         "is_domain_batch": item.get("is_domain_batch", False)
#     }

# count = 0
# with open(input_path, "r", encoding="utf-8") as fin, \
#      open(output_path, "w", encoding="utf-8") as fout:
#     for line in fin:
#         if not line.strip():
#             continue
#         data = json.loads(line)
#         converted = convert_instruction_to_messages(data)
#         fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
#         count += 1

# print(f"✅ 已处理 {count} 条数据，输出到 {output_path}")


##########################################stage-1-lightr1-sft##########################################
# from datasets import load_from_disk
# import json

# # 加载数据集
# dataset = load_from_disk("../data/stage-1-lightr1-sft")

# def correct_and_transform_sample(sample):
#     """修正角色错误并转换格式"""
#     corrected_messages = []
    
#     for i, msg in enumerate(sample["messages"]):
#         # 修正角色标注错误：第三个消息应为 assistant
#         if i == 2 and msg["role"] == "user":
#             corrected_role = "assistant"
#         else:
#             corrected_role = msg["role"]
        
#         corrected_messages.append({
#             "role": corrected_role,
#             "content": msg["content"].strip()
#         })
    
#     return {
#         "messages": corrected_messages,
#         "is_domain_batch": False
#     }

# # 处理并转换所有样本
# transformed_data = []
# for i in range(len(dataset)):
#     original_sample = dataset[i]
#     transformed = correct_and_transform_sample(original_sample)
#     transformed_data.append(transformed)

# # 保存为JSON文件
# with open("../data/stage-1-lightr1-sft_messages.jsonl", "w", encoding="utf-8") as f:
#     json.dump(transformed_data, f, ensure_ascii=False, indent=2)

# print(f"转换完成！已修正 {len(transformed_data)} 条样本中的角色错误")



# #输出数据格式
# from datasets import load_dataset

# # 加载 JSONL 数据集
# dataset_path = "../data/stage-1-lightr1-sft_messages.jsonl"
# dataset = load_dataset("json", data_files=dataset_path, split="train")  # split="train" 会返回 Dataset

# # 查看数据类型和长度
# print(f"Dataset type: {type(dataset)}")
# print(f"Number of rows: {len(dataset)}")

# # 输出第一条数据
# first_item = dataset[0]
# print("第一条数据：")
# print(first_item)


import torch
import time

def get_gpu_status(interval=1):
    """
    打印每张 GPU 的显存和利用情况（大致），不依赖 NVML。
    
    Args:
        interval: 每隔多少秒刷新一次，如果为 None，只打印一次。
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("⚠️ 当前环境没有检测到 GPU")
        return

    try:
        while True:
            print("="*40)
            for i in range(num_gpus):
                device_name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
                allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
                free_mem = reserved - allocated
                utilization = allocated / total_mem * 100  # 大致利用率（显存占用比例）

                print(f"GPU {i}: {device_name}")
                print(f"    总显存: {total_mem:.0f} MB")
                print(f"    已分配: {allocated:.0f} MB")
                print(f"    保留: {reserved:.0f} MB")
                print(f"    可用(保留-已分配): {free_mem:.0f} MB")
                print(f"    显存利用率(占总显存): {utilization:.1f}%\n")

            if interval is None:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("退出 GPU 状态监控")

if __name__ == "__main__":
    # 每 5 秒刷新一次显存占用
    get_gpu_status(interval=None)
