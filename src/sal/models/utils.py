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


import os
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

# Disable parallelism to avoid forked process warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# See model card for definitions: https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm
CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_prm_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: List[str],
    batch_size: int,
) -> List[List[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


def batched_orm_inference(
    model: PreTrainedModel, input_ids: torch.Tensor, batch_size: int
) -> List[List[float]]:
    output_scores = []
    for i in range(0, len(input_ids), batch_size):
        input_batch = input_ids[i : i + batch_size]
        with torch.no_grad():
            output_ids = model(input_batch)
            scores = output_ids.logits.float().tolist()

        output_scores.extend(scores)

        # Clear GPU memory
        del input_batch, scores
        torch.cuda.empty_cache()

    return output_scores
