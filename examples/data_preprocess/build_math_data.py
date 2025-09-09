# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import json
import random
import datasets
from datasets import Dataset

#
# sampling prob for different sources
# Source distribution in train dataset:
#     orca_math: 79556
#     cn_k12: 48974
#     big_math: 39466
#     olympiads: 26177
#     math: 8165
#     harp: 2941
#     aops_forum: 4667
#     gsm8k: 1822
#     openmath: 1650
#     omnimath: 2116
#     amc_aime: 74
SOURCE_SAMPLING_PROBS = {
    "orca_math": 0.4,
    "cn_k12": 0.45,
    "big_math": 0.5,
    "olympiads": 1.0,
    "math": 1.0,
    "harp": 1.0,
    "aops_forum": 1.0,
    "gsm8k": 1.0,
    "openmath": 1.0,
    "omnimath": 1.0,
    "amc_aime": 1.0,
}

def sample_dataset_by_source(dataset: Dataset, sampling_probs: dict, seed: int = 42) -> Dataset:
    random.seed(seed)

    def should_keep_sample(example):
        source = example['source']
        prob = sampling_probs.get(source, 1.0)  # 默认保留概率为1.0
        return random.random() < prob

    filtered_dataset = dataset.filter(should_keep_sample)
    return filtered_dataset

def stat_source_count(source_dataset):
    # stat source count
    source_counts = {}
    for example in source_dataset:
        source = example["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Source distribution in dataset(count={len(source_dataset)}):")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    return source_counts

def build_bigmatch(local_dir, random_seed):
    data_source = "open-r1/Big-Math-RL-Verified-Processed"

    raw_dataset = datasets.load_dataset(data_source, 'all')["train"].shuffle(seed=random_seed)

    # filter data by prob
    filtered_dataset = sample_dataset_by_source(raw_dataset, SOURCE_SAMPLING_PROBS, random_seed)

    split_dataset = filtered_dataset.train_test_split(test_size=0.01, seed=random_seed)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")

            user_prompt = question + " " + instruction_following

            ground_truth = example.pop("solution")

            solution = "The answer is \\boxed{" + ground_truth + "}"

            data_source = example['source']

            data = {
                "data_source": "jouw/math",
                "prompt": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    train_dataset.to_json(os.path.join(local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test.jsonl"))

    print("=" * 20, data_source, "=" * 20)

    print(f"Original dataset size: {len(raw_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Output directory: {local_dir}")

    stat_source_count(train_dataset)
    stat_source_count(test_dataset)

    # print examples
    print("=" * 20, f"{data_source} TRAIN DATASET EXAMPLES", "=" * 20)
    for i in range(min(2, len(train_dataset))):
        print(json.dumps(train_dataset[i], indent=2, ensure_ascii=False))

    print("=" * 20, f"{data_source} TEST DATASET EXAMPLES", "=" * 20)
    for i in range(min(2, len(test_dataset))):
        print(json.dumps(test_dataset[i], indent=2, ensure_ascii=False))

    return train_dataset, test_dataset

def build_math_reasoning(local_dir, random_seed):
    data_source = "notbadai/math_reasoning"

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    raw_dataset = datasets.load_dataset(data_source, 'default')["train"].shuffle(seed=random_seed)

    random.seed(random_seed)
    def should_keep_sample(example):
        return random.random() < 0.2

    filtered_dataset = raw_dataset.filter(should_keep_sample)

    split_dataset = filtered_dataset.train_test_split(test_size=0.01, seed=random_seed)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            user_prompt = question + " " + instruction_following

            reasoning = example.pop("reasoning")
            ground_truth = example.pop("answer").removesuffix("</s>")

            solution = reasoning + "\nThe answer is \\boxed{" + ground_truth + "}"

            data = {
                "data_source": "jouw/math",
                "prompt": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    train_dataset.to_json(os.path.join(local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test.jsonl"))

    print("=" * 20, data_source, "=" * 20)
    print(f"raw_dataset size: {len(raw_dataset)}")
    print(f"filtered_dataset size: {len(filtered_dataset)}")
    print(f"train_dataset size: {len(train_dataset)}")
    print(f"test_dataset size: {len(test_dataset)}")

    print("=" * 20, f"{data_source} TRAIN DATASET EXAMPLES", "=" * 20)
    for i in range(min(2, len(train_dataset))):
        print(json.dumps(train_dataset[i], indent=2, ensure_ascii=False))

    print("=" * 20, f"{data_source} TEST DATASET EXAMPLES", "=" * 20)
    for i in range(min(2, len(test_dataset))):
        print(json.dumps(test_dataset[i], indent=2, ensure_ascii=False))

    return train_dataset, test_dataset

def merge_and_save_datasets(dataset_bigmath, dataset_reasoning, output_dir, split):

    merged_dataset = datasets.concatenate_datasets([dataset_bigmath, dataset_reasoning]).shuffle(seed=42)

    # os.makedirs(output_dir, exist_ok=True)

    print(f"Merged dataset size: {len(merged_dataset)}")
    print(f"Merged dataset saved to: {output_dir}")

    merged_dataset.to_parquet(os.path.join(output_dir, f"{split}.parquet"))
    merged_dataset.to_json(os.path.join(output_dir, f"{split}.jsonl"))

    return merged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math-data")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    train_dataset_bigmath, test_dataset_bigmath = build_bigmatch(args.local_dir + "/bigmath", args.seed)
    train_dataset_reasoning, test_dataset_reasoning = build_math_reasoning(args.local_dir + "/reasoning", args.seed)

    merged_train_dataset = merge_and_save_datasets(
        train_dataset_bigmath, 
        train_dataset_reasoning, 
        args.local_dir + "/merged",
        split="train",
    )

    merged_test_dataset = merge_and_save_datasets(
        test_dataset_bigmath,
        test_dataset_reasoning,
        args.local_dir + "/merged",
        split="test",
    )

