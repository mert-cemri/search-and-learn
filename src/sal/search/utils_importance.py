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
    verification_sampling_params.prompt_logprobs = 20

    # print(f"Gen Results Before: {gen_results}")
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

        beam_index = 0
        # print(len(current_gen),len(llm_outputs)) # this is 4,4
        # assert False
        # For speculative decoding, batch process all beams through target model
        if speculative:
            # Collect all prompts with generated text for batch processing
            verification_prompts = [
                gen_result.initial_prompt + output.outputs[0].text 
                for gen_result, output in zip(current_gen, llm_outputs)
            ]
            
            # Get logprobs from target model in one batch
            verification_outputs = llm_target.generate(
                verification_prompts,
                verification_sampling_params, 
                use_tqdm=False
            )
            verification_outputs_small_model = llm.generate(
                verification_prompts,
                verification_sampling_params, 
                use_tqdm=False
            )
            # Pre-compute token ids and logprobs for each beam
            tokenizer = llm_target.get_tokenizer()
            for gen_result, output, verification_output, verification_output_small_model in zip(current_gen, llm_outputs, verification_outputs, verification_outputs_small_model):
                gen_text = output.outputs[0].text
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                
                # Calculate log probs for each token in generated text
                log_probs = []
                prompt_logprobs = verification_output.prompt_logprobs[-len(gen_token_ids):]
                prompt_logprobs_small_model = verification_output_small_model.prompt_logprobs[-len(gen_token_ids):]

                for token_id, token_logprobs in zip(gen_token_ids, prompt_logprobs):
                    if token_id in token_logprobs:
                        log_probs.append(token_logprobs[token_id].logprob)
                    else:
                        log_probs.append(-10)
                        
                gen_result.cum_prob = torch.sum(torch.tensor(log_probs))
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = output.outputs[0].stop_reason
                if gen_result.stop_reason is None:
                    gen_result.stop_reason = "EOS"

                beam_index += 1
            
        else:
            for gen_result, output in zip(current_gen, llm_outputs): # for loop is for beam_width times
                gen_text = output.outputs[0].text
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

        # print("Inside the beam search")
        # print("NEXT TEXTS:",next_texts)


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
    # print('\n\n---------------\n\n')
    # print(print(beam_result))
    # print('\n\n---------------\n\n')
    # print(len(outputs))
    
    # assert False
    return outputs

def generate_k_steps_from_next_texts(
    templated_convs, prev_next_texts, lookahead_steps, llm, sampling_params, beam_width, llm_target=None, speculative=False
):
    total_gen_tokens = []
    
    if llm_target is None:
        llm_target = llm

    gen_results = []
    assert beam_width == 1
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text+prev_next_texts[i],
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
    verification_sampling_params.prompt_logprobs = 20

    # print(f"Gen Results Before: {gen_results}")
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

        beam_index = 0
        # print(len(current_gen),len(llm_outputs)) # this is 4,4
        # assert False
        # For speculative decoding, batch process all beams through target model
        if speculative:
            # Collect all prompts with generated text for batch processing
            verification_prompts = [
                gen_result.initial_prompt + output.outputs[0].text 
                for gen_result, output in zip(current_gen, llm_outputs)
            ]
            
            # Get logprobs from target model in one batch
            verification_outputs = llm_target.generate(
                verification_prompts,
                verification_sampling_params, 
                use_tqdm=False
            )

            verification_outputs_small_model = llm.generate(
                verification_prompts,
                verification_sampling_params, 
                use_tqdm=False
            )
            
            # Pre-compute token ids and logprobs for each beam
            tokenizer = llm_target.get_tokenizer()
            for gen_result, output, verification_output, verification_output_small_model in zip(current_gen, llm_outputs, verification_outputs, verification_outputs_small_model):
                gen_text = output.outputs[0].text
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                
                # Calculate log probs for each token in generated text
                log_probs = []
                prompt_logprobs = verification_output.prompt_logprobs[-len(gen_token_ids):]
                prompt_logprobs_small_model = verification_output_small_model.prompt_logprobs[-len(gen_token_ids):]
                
                for token_id, token_logprobs, token_logprobs_small_model in zip(gen_token_ids, prompt_logprobs, prompt_logprobs_small_model):
                    if token_id in token_logprobs:
                        log_probs.append(token_logprobs[token_id].logprob - token_logprobs_small_model[token_id].logprob)
                    else:
                        log_probs.append(-10)

                total_gen_tokens.append(len(gen_token_ids))
                        
                gen_result.cum_prob = torch.sum(torch.tensor(log_probs))
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = output.outputs[0].stop_reason
                if gen_result.stop_reason is None:
                    gen_result.stop_reason = "EOS"

                beam_index += 1
            
        else:
            for gen_result, output in zip(current_gen, llm_outputs): # for loop is for beam_width times
                gen_text = output.outputs[0].text
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
            next_texts.append(prev_next_texts[i]+gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            cum_probs.append(gen_result.cum_prob)
            counter += 1

        # print("Inside the beam search")
        # print("NEXT TEXTS:",next_texts)


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
    # print('\n\n---------------\n\n')
    # print(print(beam_result))
    # print('\n\n---------------\n\n')
    # print(len(outputs))
    
    # assert False
    return outputs