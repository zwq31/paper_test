# -*- coding: utf-8 -*-
import json
import random
from collections import defaultdict

# ------------------------------
# 配置文件路径
# ------------------------------
jsonl_path = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/batch_relation_extraction/results.jsonl"

# ------------------------------
# 1️⃣ 读取文档并建立实体-文档索引
# ------------------------------
docs = []  # 保存每条文档
entity_to_doc_rels = defaultdict(list)  # 实体 -> [(doc_id, relation)]

with open(jsonl_path, "r", encoding="utf-8") as f:
    for doc_id, line in enumerate(f):
        data = json.loads(line)
        docs.append(data)
        for rel in data.get("relations", []):
            head = rel["head_entity"]
            tail = rel["tail_entity"]
            entity_to_doc_rels[head].append((doc_id, rel))
            entity_to_doc_rels[tail].append((doc_id, rel))

print(f"加载 {len(docs)} 条文档完成")

# ============================================================
# 2️⃣ 提取 2-hop 跨文档路径（每跳随机）
# ============================================================
two_hop_results = []

MAX_BC = 5  # B->C 最多选多少条

for doc_id1, doc1 in enumerate(docs):
    for rel1 in doc1.get("relations", []):
        A, B = rel1["head_entity"], rel1["tail_entity"]

        # -------- 收集 B->C 合法候选 --------
        bc_candidates = []
        for doc_id2, rel2 in entity_to_doc_rels[B]:
            if doc_id2 == doc_id1:
                continue
            if rel2["head_entity"] != B:
                continue
            bc_candidates.append((doc_id2, rel2))

        # 随机打乱
        random.shuffle(bc_candidates)

        # 截断
        for doc_id2, rel2 in bc_candidates[:MAX_BC]:
            C = rel2["tail_entity"]

            path_data = {
                "evidence_docs": [
                    {"title": docs[doc_id1]["title"]},
                    {"title": docs[doc_id2]["title"]}
                ],
                "supporting_path": [
                    A, rel1["relation"], B,
                    rel2["relation"], C
                ]
            }
            two_hop_results.append(path_data)

print(f"提取到 {len(two_hop_results)} 条 2-hop 跨文档路径")

# ============================================================
# 3️⃣ 提取 3-hop 跨文档路径（每跳随机）
# ============================================================
three_hop_results = []

MAX_BC = 2  # 第二跳 B->C
MAX_CD = 2  # 第三跳 C->D

for doc_id1, doc1 in enumerate(docs):
    for rel1 in doc1.get("relations", []):
        A, B = rel1["head_entity"], rel1["tail_entity"]

        # -------- 第二跳 B->C 候选 --------
        bc_candidates = [
            (doc_id2, rel2)
            for doc_id2, rel2 in entity_to_doc_rels[B]
            if doc_id2 != doc_id1 and rel2["head_entity"] == B
        ]
        random.shuffle(bc_candidates)

        for doc_id2, rel2 in bc_candidates[:MAX_BC]:
            C = rel2["tail_entity"]

            # -------- 第三跳 C->D 候选 --------
            cd_candidates = [
                (doc_id3, rel3)
                for doc_id3, rel3 in entity_to_doc_rels[C]
                if doc_id3 not in [doc_id1, doc_id2]
                and rel3["head_entity"] == C
            ]
            random.shuffle(cd_candidates)

            for doc_id3, rel3 in cd_candidates[:MAX_CD]:
                D = rel3["tail_entity"]

                path_data = {
                    "evidence_docs": [
                        {"title": docs[doc_id1]["title"]},
                        {"title": docs[doc_id2]["title"]},
                        {"title": docs[doc_id3]["title"]}
                    ],
                    "supporting_path": [
                        A, rel1["relation"], B,
                        rel2["relation"], C,
                        rel3["relation"], D
                    ]
                }
                three_hop_results.append(path_data)

print(f"提取到 {len(three_hop_results)} 条 3-hop 跨文档路径")

# ============================================================
# 4️⃣ 保存结果
# ============================================================
with open(
    "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/2_hop_paths.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(two_hop_results, f, ensure_ascii=False, indent=4)

with open(
    "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/3_hop_paths.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(three_hop_results, f, ensure_ascii=False, indent=4)

print("✅ 随机多跳路径提取完成！")
