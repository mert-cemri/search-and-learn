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
from dataclasses import dataclass, field

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()
import torch

import time

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

def derive_step_rewards(rewards, reward_flags):
    batch_size = rewards.shape[0]
    batch_step_rewards = []
    for i in range(batch_size):
        rewards_indices = torch.nonzero(reward_flags[i] == 1).view(-1)
        step_rewards = [rewards[i][rewards_indices[j]].item() for j in range(len(rewards_indices))]
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards

def sigmoid(x):
    return 1/(np.exp(-x) + 1)
    
def derive_step_rewards_vllm(raw_rewards, batch_reward_flags):
    batch_step_rewards = []
    for idx,data in enumerate(raw_rewards.data):
        rewards = data.embedding
        reward_flags = batch_reward_flags[idx]

        step_rewards = [sigmoid(reward) for reward,flag in zip(rewards,reward_flags) if flag == 1]   
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards

def prepare_input(problem, response, tokenizer, step_token):
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    for idx, step in enumerate(response.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

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
    cum_probs: list = field(default_factory=list)
    step_rewards: list = field(default_factory=list)


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    cum_prob: float = 0.0

def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm,
    prm,
    sampling_params: SamplingParams,
    beam_width: int,
    llm_target = None,
    speculative = False, max_tokens = 2048, config = None, draft_tokenizer=None, target_tokenizer=None, prm_tokenizer=None
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

    len_prompt_tokens = min([len(draft_tokenizer.encode(templated_convs[i])) for i in range(len(templated_convs))])
    available_tokens = max(0, config.max_context_length - len_prompt_tokens)
    gen_sampling_params = copy.deepcopy(sampling_params)
    gen_sampling_params.n = 1
    gen_sampling_params.max_tokens = min(config.max_tokens, available_tokens)
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
        # llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)

        if gen_sampling_params.max_tokens != 0:
            draft_responses = llm.completions.create(
                            model=config.draft_model_path_rsd.split("/")[-1],
                            prompt=gen_prompts,
                            temperature=config.temperature,
                            top_p=config.top_p,
                            max_tokens=gen_sampling_params.max_tokens,
                            stop=["\n\n"],
                            n=1,
                            stream=False,
                            extra_body={
                                "include_stop_str_in_output": True
                                            }
                        ).choices
            llm_outputs = sorted(draft_responses, key=lambda x: int(x.index))
        else:
            llm_outputs = []

        # print(llm_outputs)
        
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

        tokenizer = prm_tokenizer
        if gen_sampling_params.max_tokens == 0:
            for gen_result in current_gen:
                # gen_text = output.outputs[0].text
                gen_text = ''
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                
                gen_result.cum_prob = 0
                # print(f"Cumulative probability (Online serving score): {gen_result.cum_prob}")
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = "EOS"

                beam_index += 1
        elif speculative:
            # Collect all prompts with generated text for batch processing
            # verification_prompts = [
            #     gen_result.initial_prompt + output.outputs[0].text 
            #     for gen_result, output in zip(current_gen, llm_outputs)
            # ]

            ######################## BEGINNING OF VERIFICATION TARGET LOGIT RETRIEVAL ##################

            # gen_text = output.text
            # gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            
            if config.rm_regularizer != 0:
                verification_prompts = [
                    gen_result.initial_prompt + output.text 
                    for gen_result, output in zip(current_gen, llm_outputs)
                ]

                verification_outputs = llm_target.generate(
                    verification_prompts,
                    verification_sampling_params, 
                    use_tqdm=False
                )

                llm_target_tokenizer = llm_target.get_tokenizer()
            else:
                verification_outputs = llm_outputs
                llm_target_tokenizer = prm_tokenizer

            ###################### END OF VERIFICATION TARGET LOGIT RETRIEVAL ##################


            ######################### BEGIN HERE FOR PARALLEL SERVING ############################
            processed_data = [
                prepare_input(p, full_resp.text, tokenizer=prm_tokenizer, step_token="\n\n") 
                for p, full_resp in zip(gen_prompts, llm_outputs)
            ]
            input_ids, steps, reward_flags = zip(*processed_data)
            rewards = prm.embeddings.create(
                model=config.prm_path_rsd.split("/")[-1],
                input=input_ids,
            )
            step_rewards = derive_step_rewards_vllm(rewards, reward_flags)

            # THIS DOES NOT WORK BECAUSE ONLINE SERVING DOES NOT LET YOU LOOK AT EMBEDDINGS
            # target_model_logits = llm_target.embeddings.create(
            #     model=config.target_model_path_rsd.split("/")[-1],
            #     input=input_ids,
            # )


            # target_likelihoods = derive_step_rewards_vllm(target_model_logits, reward_flags)

            # batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in bad_prompts]
            # target_responses = target_client.completions.create(
            #     model=args.target_model_name_or_path.split("/")[-1],
            #     prompt=batch_prompts,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     max_tokens=args.max_tokens_per_call, 
            #     n=1,
            #     stop=[args.step_word],
            # ).choices
            # target_responses = sorted(target_responses, key=lambda x: int(x.index))
            
            # # Add target model responses to good_prompts
            # for (orig_idx, prompt, prev_responses), target_response in zip(bad_prompts, target_responses):
            #     good_prompts.append((orig_idx, prompt, prev_responses, target_response, False))  # False means using target model


            ######################### END HERE ############################

            # Pre-compute token ids and logprobs for each beam
            tokenizer = prm_tokenizer

            for gen_result, output, verification_output in zip(current_gen, llm_outputs, verification_outputs):
                # gen_text = output.outputs[0].text
                gen_text = output.text
                gen_token_ids = llm_target_tokenizer.encode(gen_text, add_special_tokens=False)

                # Calculate log probs for each token in generated text
                log_probs = []
                if config.rm_regularizer != 0:
                    prompt_logprobs = verification_output.prompt_logprobs[-len(gen_token_ids):]
                    for token_id, token_logprobs in zip(gen_token_ids, prompt_logprobs):
                        if token_id in token_logprobs:
                            log_probs.append(token_logprobs[token_id].logprob)
                        else:
                            all_current_logprobs = [token_logprobs[existing_token_id].logprob for existing_token_id in token_logprobs]
                            log_probs.append(min(all_current_logprobs))
                cum_log_probs = torch.sum(torch.tensor(log_probs))

                # print(f"Cumulative log probs: {cum_log_probs}")
                # print(f"Step rewards: {step_rewards[beam_index][-1]}")
                # log_probs = []
                # gen_result.cum_prob = torch.sum(torch.tensor(log_probs))
                if config.rm_regularizer != 0:
                    gen_result.cum_prob = cum_log_probs + config.rm_regularizer * step_rewards[beam_index][-1] # (-10, 0.85)
                else:
                    gen_result.cum_prob = step_rewards[beam_index][-1]
                # print(f"Cumulative probability (Online serving score): {gen_result.cum_prob}")
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.finish_reason
                if gen_result.first_step_stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = output.finish_reason
                # print(f"\n ******* Gen Token Length: {len(gen_token_ids)} Stop reason: {gen_result.stop_reason} ********** \n")
                # time.sleep(2)
                if gen_result.stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.stop_reason = "EOS"

                beam_index += 1
            
        else:
            tokenizer = prm_tokenizer
            for gen_result, output in zip(current_gen, llm_outputs): # for loop is for beam_width times
                gen_text = output.outputs[0].text
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
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
            cum_probs = cum_probs
        )
        outputs.append(beam_result)
    # print('\n\n---------------\n\n')
    # print(print(beam_result))
    # print('\n\n---------------\n\n')
    # print(len(outputs))
    
    # assert False
    return outputs

def generate_k_steps_from_next_texts(
    templated_convs, prev_next_texts, lookahead_steps, llm, sampling_params, beam_width, llm_target=None, speculative=False, max_tokens = 2048,
):
    if llm_target is None:
        llm_target = llm

    gen_results = []
    assert beam_width == 1
    for _, text in enumerate(templated_convs):
        for i in range(len(prev_next_texts)):
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
            
            # Pre-compute token ids and logprobs for each beam
            tokenizer = llm_target.get_tokenizer()
            for gen_result, output, verification_output in zip(current_gen, llm_outputs, verification_outputs):
                gen_text = output.outputs[0].text
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                
                # Calculate log probs for each token in generated text
                log_probs = []
                prompt_logprobs = verification_output.prompt_logprobs[-len(gen_token_ids):]
                
                for token_id, token_logprobs in zip(gen_token_ids, prompt_logprobs):
                    if token_id in token_logprobs:
                        log_probs.append(token_logprobs[token_id].logprob)
                    else:
                        log_probs.append(-10)
                        
                gen_result.cum_prob = torch.sum(torch.tensor(log_probs))
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = output.outputs[0].stop_reason
                if gen_result.stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.stop_reason = "EOS"

                beam_index += 1
            
        else:
            tokenizer = llm.get_tokenizer()
            for gen_result, output in zip(current_gen, llm_outputs): # for loop is for beam_width times
                gen_text = output.outputs[0].text
                gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.first_step_stop_reason = "EOS"

                gen_result.lookahead_text = gen_result.lookahead_text + gen_text
                gen_result.stop_reason = output.outputs[0].stop_reason
                if gen_result.stop_reason is None and len(gen_token_ids) < max_tokens:
                    gen_result.stop_reason = "EOS"

                beam_index += 1

    outputs: list[Beam] = []

    counter = 0
    
    # for _, text in enumerate(templated_convs): #happens once (or N/M times)
    next_texts = []
    stop_reasons = []
    lookahead_texts = []
    cum_probs = []
    # print(f"Len prev next texts: {len(prev_next_texts)}")
    # print(f"Len gen results: {len(gen_results)}")
    for i in range(len(prev_next_texts)):
        gen_result = gen_results[i]
        next_texts.append(prev_next_texts[i]+gen_result.first_step_text)
        lookahead_texts.append(gen_result.lookahead_text)
        stop_reasons.append(gen_result.first_step_stop_reason)
        cum_probs.append(gen_result.cum_prob)
        # counter += 1

        # print("Inside the beam search")
        # print("NEXT TEXTS:",next_texts)


    beam_result = Beam(
        prompt=templated_convs[0],
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
        cum_probs = cum_probs
    )
    outputs.append(beam_result)
    # print('\n\n---------------\n\n')
    # print(print(beam_result))
    # print('\n\n---------------\n\n')
    # print(len(outputs))
    
    # assert False
    return outputs