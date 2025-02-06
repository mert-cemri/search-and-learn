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

def _beam_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM, llm_target = None) -> list[Beam]:

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

        # # Duplicate active beams to ensure that we have config.n beams per iteration
        # if len(active_beams) != config.n:
        #     repeats = (config.n // len(active_beams)) + 1
        #     logger.debug(
        #         f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
        #     )
        #     extended_active_beams = [
        #         copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
        #     ]
        #     active_beams = extended_active_beams
        #     if len(active_beams) != config.n:
        #         raise ValueError(
        #             f"Expected {config.n} active beams, but got {len(active_beams)}"
        #         )

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
            gen_results = generate_k_steps(
                templated_convs, lookahead, llm, sampling_params, beam_width=config.n, llm_target=llm_target, speculative=False, max_tokens = config.max_tokens
            ) #1 (N/M) thing in it, with M different next_texts in each of them
        skip_sampling += 1
        if config.period > 0:
            if skip_sampling % config.period == 0:
                # next_texts = [gen_result.next_texts for gen_result in gen_results]
                prev_next_texts = gen_results[0].next_texts
                gen_results = generate_k_steps_from_next_texts(
                    templated_convs, prev_next_texts, lookahead, llm, sampling_params, beam_width=1, llm_target=llm_target, speculative=True, max_tokens = config.max_tokens
                )
             #1 (N/M) thing in it, with M different next_texts in each of them
        
            # dont_verify_flag *= -1
            # continue
        # print(f"Gen Results: {gen_results}")
        # print(f"Length of Gen Results: {len(gen_results)}")
        # # print(f"Length of Gen Results[0]: {len(gen_results[0])}")
        # print(f"Length of Gen Results[0].next_texts: {len(gen_results[0].next_texts)}")
        # assert False

        # def generate_and_select_best_beams(templated_convs, lookahead, llm, sampling_params, beam_width, m):
        #     """
        #     Generate beams using generate_k_steps and select the best n//m beams.

        #     Args:
        #         templated_convs (list): The input conversations for generation.
        #         lookahead (int): The number of lookahead steps.
        #         llm (LLM): The language model used for generation.
        #         sampling_params (SamplingParams): The parameters for sampling.
        #         beam_width (int): The number of beams to generate.
        #         m (int): The number of beams to consider.

        #     Returns:
        #         list[Beam]: The best n//m beams based on their cumulative probabilities.
        #     """
        #     gen_results = generate_k_steps(
        #         templated_convs, lookahead, llm, sampling_params, beam_width
        #     )

        #     # Sort the generated results based on cumulative probabilities
        #     sorted_results = sorted(gen_results, key=lambda x: x.cum_prob, reverse=True)
        #     # Select the top n//m beams
        #     best_beams = sorted_results[:config.n // m]
        #     return best_beams

        # # Call the function to generate and select the best beams
        # m = config.m  # Assuming config.m is defined somewhere in your code
        # best_beams = generate_and_select_best_beams(templated_convs, lookahead, llm, sampling_params, beam_width=config.n, m=m)


        # prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True): # runs for N/M (1) times
            beam.next_texts = gen_result.next_texts #new candidate beams, n amount
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.cum_prob = gen_result.cum_prob #list of probs of branches (beams), n amount
            # beam.current_text += beam.next_texts[0] #CHANGE
            # beam.history.append(beam.next_texts[0]) #CHANGE

            if (
                beam.stop_reasons[0] == "EOS"
                # or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

            tilted_scores = torch.zeros(config.n) # beam_width (M) amount of these
            prompts, completions = [], []
            cum_probs = []
            for branch_index in range(len(beam.next_texts)): # there should be beam_width (M) amount of these
                next_text = beam.next_texts[branch_index]
                # print(f"Branch Index: {branch_index}, Next Text: {next_text}")
                cum_prob = beam.cum_prob[branch_index]
                candidate = beam.current_text + next_text
                prompts.append(beam.prompt)
                completions.append([candidate])
                cum_probs.append(cum_prob)
                # score = prm._score_one_example(beam.prompt, candidate)

                # # print(f"Cum Prob: {cum_prob}")
                # # print(f"Score {score[0]}, Weighted Score: {config.rm_regularizer*score[0]}")
                # # tilted_score = cum_prob + config.rm_regularizer * score[0]

                # tilted_score = score[0]
                # # print(f"Tilted Score: {tilted_score}")
                # tilted_scores[branch_index] = tilted_score
                
            scores = prm.score(prompts, completions) # |prompts| amount of scores

            agg_scores = [
                [aggregate_scores(s, config.agg_strategy) for s in score]
                for score in scores
            ]
            
            # print(f"Scores: {len(scores)}, Len (Scores[0]): {len(scores[0])}")
            # print(f"Length of Agg Scores: {len(agg_scores)}")
            # print(f"Len (agg scores[0]): {len(agg_scores[0])}")

            # tilted_scores = agg_scores[0]
            # tilted_scores = tilted_scores - torch.max(tilted_scores)
            # probs = torch.exp(tilted_scores)/torch.sum(torch.exp(tilted_scores)) #p(x)*exp(1/beta r(x))
            
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

            tilted_scores = torch.Tensor(agg_scores).flatten() + config.rm_regularizer*torch.Tensor(cum_probs)
            tilted_scores = tilted_scores - torch.max(tilted_scores)
            probs = torch.exp(tilted_scores)/torch.sum(torch.exp(tilted_scores))
            # print(torch.Tensor(agg_scores).flatten())
            # print(torch.Tensor(cum_probs))
            # assert False
            try:
                chosen_index = torch.multinomial(probs, num_samples=1)
                # chosen_index = np.argmax(probs)
            except:
                print(f"\n Tilted Scores: {tilted_scores}")
                print(f"Probs: {probs}")
                chosen_index = 0
            # print(f"Chosen Index: {chosen_index}")
            beam.current_text += beam.next_texts[chosen_index]
            # print(f"Chosen Text: {beam.next_texts[chosen_index]}")
            # print(f"Current Text: {beam.current_text}")
            if beam.all_scores:
                beam.all_scores.append([tilted_scores[chosen_index]])
            else:
                beam.all_scores = [[tilted_scores[chosen_index]]]
            beam.history.append(beam.next_texts[chosen_index])
            # prompts.append(beam.prompt)
            # completions.append([beam.current_text])

        # scores = prm.score(prompts, completions)

        # agg_scores = [
        #     [aggregate_scores(s, config.agg_strategy) for s in score]
        #     for score in scores
        # ]

        # for beam, score in zip(active_beams, scores, strict=True):
        #     beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        # agg_scores = [
        #     agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        # ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            # print(f"\nStopping reason: {beam.stop_reasons[0]}\n\n")
            break

        # # Filter duplicate active beams
        # if config.filter_duplicates:
        #     # Create a dictionary to filter duplicates and retain order
        #     unique_beam_dict = {}
        #     for i, b in enumerate(active_beams):
        #         if b.current_text not in unique_beam_dict:
        #             unique_beam_dict[b.current_text] = (
        #                 i  # Map the unique text to its index
        #             )
        #     active_beams = [active_beams[i] for i in unique_beam_dict.values()]
        #     agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # # Get indices for top (config.n / config.beam_width) completions
        # top_indices = np.argsort(np.array(agg_scores).flatten())[
        #     -(config.n // config.n) :
        # ]

        # for idx, beam in enumerate(active_beams):
        #     if idx not in top_indices:
        #         beam.pruned = True


        # print("+++++++++++++++++++++++\n\n")
        # print(f"NUMBER OF ACTIVE BEAMS: {len(active_beams)}")
        # print("+++++++++++++++++++++++\n\n")

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

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


def beam_search(examples, config: Config, llm: LLM, prm: PRM, llm_target=None):
    problems = examples["problem"]
    beam_results = []
    try:
        beam_results = _beam_search(problems, config, llm, prm, llm_target)
    except Exception as e:
        print(f"\n\n *** An error occurred: {e} *** \n\n")

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

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

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
        # agg_scores = [
        #     aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        # ]
        pred = completions[0]
        # print(f"Prompt: {p}")
        # print(f"Completions: {completions}")
        # print(f"Prediction: {pred}")
        # assert False
        results["completions"].append(completions)
        results["scores"].append([b.cum_prob for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
