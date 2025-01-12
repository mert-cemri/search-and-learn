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
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts, speculative_beam_search
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "speculative_beam_search": speculative_beam_search,
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        max_model_len=config.max_model_len,
    )
    prm = load_prm(config)        

    dataset = get_dataset(config)

    import numpy as np
    # def process_example(example):
    # # Ensure all float values are of the same dtype
    #     example["column_name"] = np.array(example["column_name"], dtype=np.float32)
    #     return example
    # dataset = dataset.map(process_example)

    if config.approach == "speculative_beam_search":
        llm_target = LLM(
        model=config.target_model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        max_model_len=config.max_target_model_len,
        )
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm, "llm_target": llm_target},
            desc="Running search",
            load_from_cache_file=False,
        )
    else:
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )
    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
