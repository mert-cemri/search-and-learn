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


@dataclass
class Config:
    approach: str = "beam_search"
    model_path: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"

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
    prm_batch_size: int = 16  # Larger batch sizes can lead to OOM errors with the PRM/ORM server
    search_batch_size: int = 16
    
    # Best of N search options
    # TODO
    # Beam Search options
    beam_search_n_iters: int = 40
    beam_search_width: int = 4  # m in the paper
    
    def __post_init__(self):
        print("This is the BeamSearchConfig class.")
        
        if self.approach == "beam_search":
        
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")

            self.n_beams = self.n // self.beam_width    
    
    
