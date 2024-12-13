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

from dataclasses import dataclass

from huggingface_hub import (
    create_branch,
    get_full_repo_name,
    list_repo_commits,
    repo_exists,
)

from sal.utils.hub import get_dataset_revisions


@dataclass
class Config:
    approach: str = (
        "parallel_beamsearch"  # Options: "beam_search", "parallel_beamsearch", "best_of_n"
    )
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    gpu_memory_utilization: float = 0.5 # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None

    # Search Related Options
    n: int = 4
    temperature: float = 1.0
    prm_batch_size: int = (
        16  # Larger batch sizes can lead to OOM errors with the PRM/ORM server
    )
    search_batch_size: int = 16
    seed: int = 42

    # Best of N search options
    # TODO
    # Beam Search options
    beam_search_n_iters: int = 40
    beam_search_width: int = 4  # m in the paper

    def __post_init__(self):
        if self.approach == "beam_search":
            if self.n % self.beam_search_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_search_width

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                # Set default based on model name. We prepend the username for compatibility with the repo checks below.
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach == "beam_search":
                revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}--look-{self.lookahead}--seed-{self.seed}--agg_strategy-last"
            else:
                raise ValueError(f"Unknown approach {self.approach}")
            if self.dataset_start is not None and self.dataset_end is not None:
                revision = f"{revision}--chunk-{self.dataset_start}_{self.dataset_end}"

            # Early exit if the revision on the Hub already exists
            if not self.overwrite_hub_revision and revision in revisions:
                # logger.info(f"Revision {revision} already exists on the Hub. Exiting.")
                exit()
