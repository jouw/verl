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
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math_reasoning_refine")

    args = parser.parse_args()

    data_source = "notbadai/math_reasoning"

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    raw_dataset = datasets.load_dataset(data_source, 'default')["train"].shuffle(seed=42)
    split_dataset = raw_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            user_prompt = question + " " + instruction_following

            ground_truth = example.pop("answer").removesuffix("</s>")

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
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    train_dataset.to_json(os.path.join(args.local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(args.local_dir, "test.jsonl"))

    print(f"train_dataset size: {len(train_dataset)}")
    print(f"test_dataset size: {len(test_dataset)}")
    print(f"output directory: {args.local_dir}")

    print("=" * 20, "TRAIN DATASET EXAMPLES", "=" * 20)
    for i in range(min(3, len(train_dataset))):
        print(json.dumps(train_dataset[i], indent=2, ensure_ascii=False))

    print("=" * 20, "TEST DATASET EXAMPLES", "=" * 20)
    for i in range(min(3, len(test_dataset))):
        print(json.dumps(test_dataset[i], indent=2, ensure_ascii=False))
