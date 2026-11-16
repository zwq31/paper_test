import json
from pathlib import Path

# 配置路径
EVAL_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/test_decision_tree_2400/evaluation_results.json"      # 第一类：有 evaluation_results
DECISION_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/test_decision_tree_2400.json"    # 第二类：有 should_answer
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/test_decision_tree_2400/merged_evaluation.json"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_by_question(evaluation_data, decision_data):
    # 构建 question -> data 的映射
    eval_map = {item["question"]: item for item in evaluation_data}
    decision_map = {item["question"]: item for item in decision_data}

    merged = []
    all_questions = set(eval_map.keys()) | set(decision_map.keys())

    for q in all_questions:
        eval_item = eval_map.get(q, {})
        decision_item = decision_map.get(q, {})

        # 检查 source_id 是否一致（可选）
        source_id_eval = eval_item.get("source_id")
        source_id_dec = decision_item.get("source_id")
        if source_id_eval is not None and source_id_dec is not None and source_id_eval != source_id_dec:
            print(f"⚠️ Warning: source_id mismatch for question: {q}")

        # 合并
        merged_item = {
            "source_id": source_id_eval or source_id_dec,
            "question": q,
            "answer": eval_item.get("answer", ""),
            "model_answer": eval_item.get("model_answer") or decision_item.get("model_answer", ""),
            # 来自 decision
            "refusal_decision": {
                "should_answer": decision_item.get("should_answer", True),
                "decision_metrics": decision_item.get("decision_metrics", {}),
                "decision_details": decision_item.get("decision_details", {})
            },
                        # 来自 evaluation
            "evaluation_results": eval_item.get("evaluation_results", {}),
        }

        merged.append(merged_item)

    return merged

if __name__ == "__main__":
    eval_data = load_json(EVAL_FILE)
    decision_data = load_json(DECISION_FILE)

    merged_data = merge_by_question(eval_data, decision_data)

    # 保存合并结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成！共 {len(merged_data)} 条样本，已保存至: {OUTPUT_FILE}")