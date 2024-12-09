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


from itertools import accumulate

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.utils import get_prm_model_and_tokenizer

from .utils import batched_prm_inference, get


def get_prm_model_and_tokenizer():
    model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # For batched inference
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    ).eval()
    return model, tokenizer


class PRM:
    def __init__(self, config: Config):
        self.beam_config = config
        self.model, self.tokenizer = get_prm_model_and_tokenizer()
        # self.tokenizer.pad_token = self.tokenizer.eos_token  # For batched inference

    def score(self, questions: list[str], outputs: list[list[str]]):
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.beam_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n", " ки\n") for o in output]
            special_outputs = [
                o + " ки" if o[-1] != "\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_prm_inference(
            self.model, self.tokenizer, inputs_for_prm, self.beam_config.prm_batch_size
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores
