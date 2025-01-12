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

from .utils import Beam, build_conv, generate_k_steps, last

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


    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
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
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, beam_width=config.n, llm_target=llm_target, speculative=True
        ) #1 (N/M) thing in it, with M different next_texts in each of them

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
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

            tilted_scores = torch.zeros(config.n) # beam_width (M) amount of these
            for branch_index in range(len(beam.next_texts)): # there should be beam_width (M) amount of these
                next_text = beam.next_texts[branch_index]
                # print(f"Branch Index: {branch_index}, Next Text: {next_text}")
                cum_prob = beam.cum_prob[branch_index]
                candidate = beam.current_text + next_text
                score = prm._score_one_example(beam.prompt, candidate)
                # print(f"Cum Prob: {cum_prob}")
                tilted_score = cum_prob + config.rm_regularizer * score[0]
                # print(f"Tilted Score: {tilted_score}")
                tilted_scores[branch_index] = tilted_score

            probs = torch.exp(tilted_scores)/torch.sum(torch.exp(tilted_scores)) #p(x)*exp(1/beta r(x))
            try:
                chosen_index = torch.multinomial(probs, num_samples=1)
            except:
                print(f"Tilted Scores: {tilted_scores}")
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
    beam_results = _beam_search(problems, config, llm, prm, llm_target)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        
        assert len(beams)==1

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
