# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm  # 进度条支持

from categories import categories, subcategories

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    # Step 1: Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
        trust_remote_code=True
    )

    # # Step 2: Load LoRA adapter
    # if args.lora_path:
    #     model = PeftModel.from_pretrained(model, args.lora_path)
    #     print(f"✅ Loaded LoRA weights from {args.lora_path}")
    # else:
    #     print("⚠️ No LoRA path provided, using base model only.")

    model.eval()

    # Step 3: Get subjects
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # Step 4: Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    results_subdir = os.path.join(args.save_dir, "results_{}".format(args.model_name))
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    # Step 5: Evaluate each subject with progress bar
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    print(f"🚀 Start time: {datetime.datetime.now()}")

    with tqdm(total=len(subjects), desc="Evaluating MMLU Subjects", ncols=100, unit="subject") as pbar:
        for subject in subjects:
            dev_df = pd.read_csv(
                os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
            )[: args.ntrain]
            test_df = pd.read_csv(
                os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
            )

            cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
            subcats = subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(cors)
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key].append(cors)
            all_cors.append(cors)

            test_df["{}_correct".format(args.model_name)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(args.model_name, choice)] = probs[:, j]
            test_df.to_csv(
                os.path.join(results_subdir, "{}.csv".format(subject)),
                index=None,
            )

            pbar.update(1)
            pbar.set_postfix_str(f"Last: {subject} | Acc: {acc:.3f}")

    # Step 6: Save results
    results = {"subcategories": {}, "categories": {}, "start_time": datetime.datetime.now().isoformat()}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Overall Average accuracy: {:.3f}".format(weighted_acc))

    # Save JSON results
    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model_name.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    results["end_time"] = datetime.datetime.now().isoformat()
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))

    print(f"✅ End time: {datetime.datetime.now()}")
    print(f"⏱️ Total cost time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/mnt/project/Qwen2.5-7B-Instruct-news/general_test_dataset/MMLU")
    parser.add_argument("--save_dir", "-s", type=str, default="/mnt/project/Qwen2.5-7B-Instruct-news/general_test_dataset/MMLU_result/Qwen2.5-7B-Instruct-CPT-3years-QLORA-1year-2dataset")

    # 模型相关参数
    parser.add_argument("--base_model", "-bm", type=str, default="/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output-cpt-drouput0.1-r128-bs16-epoch5/final_model_2dataset", help="基础模型路径")
    # parser.add_argument("--lora_path", "-lp", type=str, default="../../output-continued-pretraining/3years/checkpoint-1980", help="LoRA 权重路径")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="是否使用 4bit 推理加速（默认开启）")
    parser.add_argument("--model_name", "-mn", type=str, default="qwen2.5-7b-instruct-mol-qlora", help="用于结果文件命名的模型标识符")

    args = parser.parse_args()
    main(args)