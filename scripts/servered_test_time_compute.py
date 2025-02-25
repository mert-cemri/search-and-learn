#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging

import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm, MergedModel
from sal.search import beam_search, best_of_n, dvts, speculative_beam_search, speculative_importance_search, speculative_beam_search_merged_models, online_speculative_beam_search
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score


import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "speculative_importance_search": speculative_importance_search,
    "speculative_beam_search": speculative_beam_search,
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "speculative_beam_search_merged_models": speculative_beam_search_merged_models,
    "online_speculative_beam_search": online_speculative_beam_search,
}

from openai import OpenAI
from transformers import AutoTokenizer

# from external.qwen25_math_evaluation.evaluate import evaluate
# from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
# from external.qwen25_math_evaluation.parser import *
# from external.qwen25_math_evaluation.trajectory import *
# from external.qwen25_math_evaluation.data_loader import load_data
# from external.qwen25_math_evaluation.python_executor import PythonExecutor
# from external.skywork_o1_prm_inference.model_utils.io_utils import prepare_input, derive_step_rewards_vllm

def setup(args):
    # load model
    openai_api_key = "EMPTY"
    draft_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.draft_model_path_rsd,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model_path_rsd, trust_remote_code=True)

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_model_path_rsd,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path_rsd, trust_remote_code=True)

    prm_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.prm_ip_address_rsd,
    )
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_path_rsd, trust_remote_code=True)

    return draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer

    # # infer & eval
    # data_list = args.data_names.split(",")
    # results = []
    # for data_name in data_list:
    #     results.append(main(draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, data_name, args))

    # # add "avg" result to data_list and results
    # data_list.append("avg")
    # results.append({"acc": sum([result["acc"] for result in results]) / len(results),})

    # # print all results
    # pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer = setup(config)

    assert config.approach == "online_speculative_beam_search"
    approach_fn = APPROACHES[config.approach]

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
         

    dataset = get_dataset(config)

    import numpy as np
    # def process_example(example):
    # # Ensure all float values are of the same dtype
    #     example["column_name"] = np.array(example["column_name"], dtype=np.float32)
    #     return example
    # dataset = dataset.map(process_example)

    
    # if config.approach == "speculative_beam_search" or config.approach =="speculative_importance_search":
    #     llm_target = LLM(
    #     model=config.target_model_path,
    #     gpu_memory_utilization=config.target_gpu_memory_utilization,
    #     enable_prefix_caching=False,
    #     seed=config.seed,
    #     tensor_parallel_size=num_gpus,
    #     max_model_len=config.max_target_model_len,
    #     )

    # if config.approach == "speculative_beam_search_merged_models":
    #     from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
    #     base_model_path = "../models_merged/Qwen2--qwen_7b_merged"
    #     reward_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
    #     causal_model_name = "Qwen/Qwen2.5-7B-Instruct"

    #     # Load tokenizer
    #     # from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")
    #     # Load model (automatically detects and uses safetensors)
    #     base_model = AutoModel.from_pretrained(base_model_path).eval()

    #     reward_model = AutoModel.from_pretrained(
    #         reward_model_name, 
    #         trust_remote_code=True,
    #     ).eval()
    #     causal_model = AutoModelForCausalLM.from_pretrained(
    #         causal_model_name,
    #     ).eval()

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     merged_model_reward = MergedModel(tokenizer=tokenizer, base_model=base_model, reward_model=reward_model, 
    #                        causal_model=causal_model, beginning_offset=9, total_tokens=29, device=device)
    # else:
    #     prm = load_prm(config)

# draft_responses = draft_client.completions.create(
#             model=args.draft_model_name_or_path.split("/")[-1],
#             prompt=batch_prompts,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             max_tokens=args.max_tokens_per_call,
#             stop=[args.step_word],
#         ).choices
#         draft_responses = sorted(draft_responses, key=lambda x: int(x.index))
    start = time.time()
    dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": draft_client, "prm": prm_client, "llm_target": target_client},
            desc="Running (speculative) search",
            load_from_cache_file=False,
        )

    end = time.time()
    runtime = end - start

    print(f"Runtime: {runtime}")

    dataset = score(dataset, config)

    # # Add runtime to dataset
    # dataset = dataset.add_column("runtime", [runtime] * len(dataset))

    save_dataset(dataset, config)
    logger.info("Done 🔥!")
    logger.info(f"Runtime: {runtime}")

if __name__ == "__main__":
    main()
