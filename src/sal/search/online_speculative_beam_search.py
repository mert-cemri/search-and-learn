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
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last, generate_k_steps_from_next_texts

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

import torch

import time

def _beam_search(batch_of_prompts, config, draft_client, prm: PRM, llm_target = None) -> list[Beam]:

    if llm_target is None:
        llm_target = llm

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=config.n,
        logprobs=1
    )

    draft_responses = draft_client.completions.create(
                model=config.draft_model_name_or_path.split("/")[-1],
                prompt=batch_of_prompts,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens_per_call,
                stop=[config.step_word],
            ).choices
    draft_responses = sorted(draft_responses, key=lambda x: int(x.index))
    # Evaluate draft responses with PRM
    full_responses = [''.join(r[0] for r in prev_resp) + new_resp.text
                for (_, _, prev_resp), new_resp in zip(current_prompts, draft_responses)]
    processed_data = [
        prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word) 
        for p, full_resp in zip(current_problems, full_responses)
    ]
    input_ids, steps, reward_flags = zip(*processed_data)
    rewards = prm_client.embeddings.create(
        input=input_ids,
        model=args.prm_name_or_path.split("/")[-1],
    )
    step_rewards = derive_step_rewards_vllm(rewards, reward_flags) # list[list]
    # Split prompts based on step_reward
    good_prompts = []
    bad_prompts = []
    for (orig_idx, prompt, prev_responses), draft_response, step_reward in zip(current_prompts, draft_responses, step_rewards):
        all_rewards[orig_idx].append(round(step_reward[-1], 6))
        if step_reward[-1] >= args.prm_threshold:
            good_prompts.append((orig_idx, prompt, prev_responses, draft_response, True))  # True means using draft model
        else:
            draft_response_text = draft_response.text + args.step_word
            token_counts[orig_idx] = (
                token_counts[orig_idx][0], 
                token_counts[orig_idx][1], 
                token_counts[orig_idx][2]+len(draft_tokenizer.encode(draft_response_text))
            )
            bad_prompts.append((orig_idx, prompt, prev_responses))

    # Generate using target model for bad prompts
    if bad_prompts:
        batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in bad_prompts]
        target_responses = target_client.completions.create(
            model=args.target_model_name_or_path.split("/")[-1],
            prompt=batch_prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call, 
            n=1,
            stop=[args.step_word],
        ).choices
        target_responses = sorted(target_responses, key=lambda x: int(x.index))
        
        # Add target model responses to good_prompts
        for (orig_idx, prompt, prev_responses), target_response in zip(bad_prompts, target_responses):
            good_prompts.append((orig_idx, prompt, prev_responses, target_response, False))  # False means using target model


    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        i = 0
        # for i in range(config.n):
        beams.append(
            Beam(
                prompt=prompt,
                index=i,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                pruned=False,
                completed=False,  # New flag to track completion
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=0,
            )
        )

    completed_beams: list[Beam] = []

    verify_flag = -1
    skip_sampling = 0
    for i in tqdm(range(config.num_iterations), desc="Speculative beam search iterations"):
        # print(i)
        # assert False
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]

        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        

        if config.period == 0:
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, beam_width=config.n, llm_target=llm_target, speculative=True, max_tokens = config.max_tokens
            ) #1 (N/M) thing in it, with M different next_texts in each of them
        else:
            if skip_sampling % config.period == 0:
                gen_results = generate_k_steps(
                    templated_convs, lookahead, llm, sampling_params, beam_width=config.n, llm_target=llm_target, speculative=False, max_tokens = config.max_tokens
                ) #1 (N/M) thing in it, with M different next_texts in each of them
                prev_next_texts = gen_results[0].next_texts
                gen_results = generate_k_steps_from_next_texts(
                    templated_convs, prev_next_texts, lookahead, llm, sampling_params, beam_width=1, llm_target=llm_target, speculative=True, max_tokens = config.max_tokens
                )
            else:
                gen_results = generate_k_steps(
                    templated_convs, lookahead, llm, sampling_params, beam_width=config.n, llm_target=llm_target, speculative=True, max_tokens = config.max_tokens
                ) #1 (N/M) thing in it, with M different next_texts in each of them
        skip_sampling += 1

        for beam, gen_result in zip(active_beams, gen_results, strict=True): # runs for N/M (1) times
            beam.next_texts = gen_result.next_texts #new candidate beams, n amount
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.cum_probs = gen_result.cum_probs #list of probs of branches (beams), n amount
            # beam.current_text += beam.next_texts[0] #CHANGE
            # beam.history.append(beam.next_texts[0]) #CHANGE

            if (
                beam.stop_reasons[0] == "EOS"
                # or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
                or i == config.num_iterations - 1
            ):
                beam.completed = True
                completed_beams.append(beam)

            tilted_scores = torch.zeros(config.n) # beam_width (M) amount of these
            prompts, completions = [], []
            cum_probs = []
            for branch_index in range(len(beam.next_texts)): # there should be beam_width (M) amount of these
                next_text = beam.next_texts[branch_index]
                # print(f"Branch Index: {branch_index}, Next Text: {next_text}")
                cum_prob = beam.cum_probs[branch_index]
                candidate = beam.current_text + next_text
                prompts.append(beam.prompt)
                completions.append([candidate])
                cum_probs.append(cum_prob)

            scores = prm.score(prompts, completions) # |prompts| amount of scores
            # print(f"\nScores: {scores}\n")
            # time.sleep(2)
            agg_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in scores
            ]

            """
            Scores: [[[1.0, 1.0, 0.9609375]], [[1.0, 1.0, 0.99609375]], [[0.1640625, 0.2021484375]], [[0.6796875, 0.99609375]], [[0.99609375, 1.0, 0.953125]], [[1.0, 1.0, 0.98828125]], [[0.99609375, 1.0, 0.97265625]], [[0.99609375, 1.0, 0.95703125]]]
            Agg Scores: [[0.9609375], [0.99609375], [0.2021484375], [0.99609375], [0.953125], [0.98828125], [0.97265625], [0.95703125]]
            Scores: 8, Len (Scores[0]): 1
            Length of Agg Scores: 8
            Len (agg scores[0]): 1
            """

            # Filter duplicate active beams

            if config.filter_duplicates:
                # Create a dictionary to filter duplicates and retain order
                unique_beam_dict = {}
                for i, candidate in enumerate(beam.next_texts):
                    if candidate not in unique_beam_dict:
                        unique_beam_dict[candidate] = (
                            i  # Map the unique text to its index
                        )
                agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]
                cum_probs = [cum_probs[i] for i in unique_beam_dict.values()]

            cum_probs_tensor = torch.Tensor(cum_probs)
            agg_scores_tensor = torch.Tensor(agg_scores)
            assert agg_scores_tensor.flatten().shape == cum_probs_tensor.flatten().shape
            og_tilted_scores = agg_scores_tensor.flatten() + config.rm_regularizer*cum_probs_tensor.flatten()
            tilted_scores = og_tilted_scores - torch.max(og_tilted_scores)
            probs = torch.exp(tilted_scores)/torch.sum(torch.exp(tilted_scores))

            try:
                chosen_index = torch.multinomial(probs, num_samples=1)
            except:
                print(f"\n Tilted Scores: {tilted_scores}")
                print(f"Probs: {probs}")
                chosen_index = 0
            beam.current_text += beam.next_texts[chosen_index]

            if beam.all_scores:
                beam.all_scores.append(og_tilted_scores[chosen_index].item())
            else:
                beam.all_scores = [og_tilted_scores[chosen_index].item()]
            beam.history.append(beam.next_texts[chosen_index])

        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

    

    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    # print(f"\n\n All Scores (Final): {[b.all_scores for b in completed_beams]} \n")

    # if len(completed_beams) != config.n:
    #     # If we don't have enough completed_beams, duplicate until we reach config.n
    #     repeats = (config.n // len(completed_beams)) + 1
    #     logger.debug(
    #         f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
    #     )
    #     extended_completed_beams = [
    #         copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
    #     ]
    #     completed_beams = extended_completed_beams

    return completed_beams


def beam_search(examples, config, llm=None, prm=None, llm_target=None):
    # try:
        problems = examples["problem"]
        beam_results = []
        start_time = time.time()
        # try:
        beam_results = _beam_search(problems, config, llm, prm, llm_target)
        # except Exception as e:
        #     print(f"\n\n *** An error occurred while running beam search: {e} *** \n\n")
        end_time = time.time()
        time_taken = end_time - start_time
        if len(beam_results)==0:
            beam_results: list[Beam] = []
            for prompt in problems:
                i = 0
                # for i in range(config.n):
                beam_results.append(
                    Beam(
                        prompt=prompt,
                        index=i,
                        current_text="",
                        next_texts=None,
                        lookahead_texts=None,
                        pruned=False,
                        completed=False,  # New flag to track completion
                        stop_reasons=None,
                        history=[],
                        best_scores=[],
                        all_scores=[],
                        previous_text=None,
                        completion_tokens=0,
                    )
                )

        # Group together alike beams and store in the dataset
        grouped_results = defaultdict(list)
        for results in beam_results:
            grouped_results[results.prompt].append(results)

        results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "runtime": [time_taken] * len(problems)}

        for p in problems:
            beams = grouped_results[p]
            
            if len(beams)!=1:
                print(f"Number of beams: {len(beams)}")
                for beam in beams:
                    print(f"Beam: {beam.current_text}")
            # else:
            #     print(f"Number of beams: {len(beams)}")
            assert len(beams)<=1

            completions = [b.current_text for b in beams]
            try:
                agg_scores = [
                    aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
                ]
            except Exception as e:
                print(f"Number of beams: {len(beams)}")
                print(f"\n\n *** An error occurred at the last stage of beam search: {e} *** \n\n")
                print(f"All Scores: {[b.all_scores for b in beams]}")
                # assert False
            pred = completions[0]
            # print(f"Prompt: {p}")
            # print(f"Completions: {completions}")
            # print(f"Prediction: {pred}")
            # assert False
            results["completions"].append(completions)
            # print(f"Cum Probs: {[b.cum_probs for b in beams]}")
            # print(f"Beam scores agg score example: {agg_scores[0]}")
            # print(f"All Scores: {[b.all_scores for b in beams]}")
            results["scores"].append([b.all_scores for b in beams])
            results["pred"].append(pred)
            results["completion_tokens"].append([b.completion_tokens for b in beams])

        return results
    # except Exception as e:
    #     print(f"\n\n *** An error occurred at the last stage of beam search: {e} *** \n\n")
    #     results ={
    #         "completions": [["Error occurred"] for _ in problems],
    #         "pred": ["Error occurred" for _ in problems],
    #         "completion_tokens": [[0] for _ in problems],
    #         "scores": [[[0.0]] for _ in problems],
    #         "runtime": [0] * len(problems)
    #     }
    #     return results
    