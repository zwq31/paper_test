import json
import math
import os

class DiscreteEntropyProcessor:
    def __init__(self, threshold=0.5):
        """
        threshold: 判定幻觉的阈值。
        根据论文和实验经验，DSE > 0.5 通常被视为存在语义分歧。
        """
        self.threshold = threshold

    def calculate_dse(self, clustering_details):
        """计算离散语义熵"""
        clusters = clustering_details.get("clusters", [])
        total_samples = sum(c["size"] for c in clusters)
        
        if total_samples == 0:
            return 0.0
        
        dse = 0.0
        for cluster in clusters:
            p = cluster["size"] / total_samples
            if p > 0:
                dse -= p * math.log(p)
        
        return dse

    def process_json_streaming(self, input_path, output_path):
        if not os.path.exists(input_path):
            print(f"错误: 找不到文件 {input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 确保 data 是列表
        if isinstance(data, dict):
            data = [data]

        written_count = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("[\n")
            f_out.flush()

            for entry in data:
                try:
                    dse = self.calculate_dse(entry["clustering_details"])
                    entry["discrete_semantic_entropy"] = round(dse, 6)
                    entry["is_hallucination_dse"] = bool(dse > self.threshold)

                    # 序列化当前条目
                    json_str = json.dumps(entry, ensure_ascii=False, indent=2)
                    indented_lines = ["  " + line for line in json_str.splitlines()]

                    if written_count > 0:
                        f_out.write(",\n")
                    f_out.write("\n".join(indented_lines))
                    f_out.flush()
                    written_count += 1

                except Exception as e:
                    print(f"⚠️ 跳过一条异常数据: {e}")
                    continue

            f_out.write("\n]\n")
            f_out.flush()

        print(f"✅ 处理完成！")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        print(f"总处理条数: {written_count}")
        print(f"判定阈值: {self.threshold}")


# ==================== 执行脚本 ====================
if __name__ == "__main__":
    INPUT_FILE = "/mnt/project/semantic_entropy/data/train_semantic_entropy.json"
    OUTPUT_FILE = "/mnt/project/semantic_entropy/data/train_discrete_entropy.json"

    processor = DiscreteEntropyProcessor(threshold=0.5)
    processor.process_json_streaming(INPUT_FILE, OUTPUT_FILE)