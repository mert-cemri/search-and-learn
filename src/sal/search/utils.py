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
import copy
import logging
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()
import torch

def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]] | None # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0
    cum_prob: int = 0


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    cum_prob: int = 0

def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
    llm_target = None,
    speculative = False,
) -> list[Beam]:
    if llm_target is None:
        llm_target = llm

    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)
    gen_sampling_params.n = 1
    verification_sampling_params = copy.deepcopy(sampling_params)
    verification_sampling_params.n=1 #dont generate anything when verifying
    verification_sampling_params.max_tokens=1  #dont generate anything when verifying
    verification_sampling_params.prompt_logprobs = 1

    #what is the purpose of lookahead_steps?
    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results)) # beam width 
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ] #4 prompts, essentially
        # print(gen_prompts[0])
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        # assert False
        # print('-------------\n')
        # print(llm_outputs[0].outputs[0])
        # print('-------------\n')
        # print(len(llm_outputs[0].outputs))
        # print('-------------\n')
        # assert False
        tokenizer = llm.get_tokenizer()
        token_ids = tokenizer.encode(gen_result.initial_prompt, add_special_tokens=True) 

        beam_index = 0
        # print(len(current_gen),len(llm_outputs)) # this is 4,4
        # assert False
        for gen_result, output in zip(current_gen, llm_outputs): # for loop is for beam_width times

            print("DOING BEAM:",beam_index)

            gen_text = output.outputs[0].text #why zero here? should it be beam_index?
            # gen_result.cum_prob = output.outputs[0].cumulative_logprob
            if speculative:
                prompt_to_be_done = gen_result.initial_prompt
                # print("\n\n----\n\n ",prompt_to_be_done,"\n\n--------\n\n")
                # print(len(gen_prompts))
                # print("--------\n\n")
                # assert False
                new_answer = prompt_to_be_done + gen_text
                verification_output = llm_target.generate(new_answer, verification_sampling_params, use_tqdm=False)
                # assert False
                # logprobs = torch.sum([logprob.logprob for logprob in verification_output[0].prompt_logprobs[-gen_sampling_params.max_tokens:]])
                keys = token_ids[-gen_sampling_params.max_tokens:]
                counter_start = -len(keys)
                # log_probs = [verification_output[0].prompt_logprobs[counter_start + i].get(key).logprob if verification_output[0].prompt_logprobs[counter_start + i].get(key) else 0.0 for i, key in enumerate(keys)]
                log_probs = []
                for i,key in enumerate(keys):
                    if verification_output[0].prompt_logprobs[counter_start + i]:
                        lp = verification_output[0].prompt_logprobs[counter_start + i].get(key).logprob
                    else:
                        lp = 0
                    log_probs.append(lp)
                # print("Log probs:")
                # print(log_probs)
                # assert False
                total_log_probs = torch.sum(torch.tensor(log_probs))
                gen_result.cum_prob = total_log_probs
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

            beam_index += 1

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs): #happens once (or N/M times)
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        cum_probs = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            cum_probs.append(gen_result.cum_prob)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            cum_prob = cum_probs
        )
        outputs.append(beam_result)
    print('\n\n---------------\n\n')
    print(print(beam_result))
    print('\n\n---------------\n\n')
    print(len(outputs))
    
    # assert False
    return outputs