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
import re
import json
import random
import datasets
from datasets import Dataset

def get_math():
    def remove_boxed(s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left):]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left): -1]

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]

        return retval

    def extract_solution(solution_str):
        return remove_boxed(last_boxed_only_string(solution_str))

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    print(f"{data_source} train_dataset count: {len(train_dataset)}")
    print(f"{data_source} test_dataset count: {len(test_dataset)}")
    return train_dataset, test_dataset

def get_gsm8k():

    def extract_solution(solution_str):
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        assert solution is not None
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
        return final_solution

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    print(f"{data_source} train_dataset count: {len(train_dataset)}")
    print(f"{data_source} test_dataset count: {len(test_dataset)}")
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math-data-v1")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()


    train_math, test_math = get_math()
    train_gsm8k, test_gsm8k = get_gsm8k()

    def make_map_fn(split):
        def process_fn(example, idx):
            example["extra_info"]["index"] = idx
            return example
        return process_fn

    train_dataset = datasets.concatenate_datasets([train_math, train_gsm8k])
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset.map(function=make_map_fn("train"), with_indices=True)

    test_dataset = datasets.concatenate_datasets([test_math, test_gsm8k])
    test_dataset = test_dataset.shuffle(seed=42)
    test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    train_dataset.to_json(os.path.join(args.local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(args.local_dir, "test.jsonl"))

    print(f"final train_dataset count: {len(train_dataset)}")
    print(f"final test_dataset count: {len(test_dataset)}")

    # print examples
    print("=" * 20, f"TRAIN DATASET EXAMPLES", "=" * 20)
    for i in range(min(3, len(train_dataset))):
        print(json.dumps(train_dataset[i], indent=2, ensure_ascii=False))

    print("=" * 20, f"TEST DATASET EXAMPLES", "=" * 20)
    for i in range(min(3, len(test_dataset))):
        print(json.dumps(test_dataset[i], indent=2, ensure_ascii=False))



