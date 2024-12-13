
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

import logging
from typing import Any

from vllm import LLM, SamplingParams
import numpy as np
from sal.config import Config
from sal.models.prm import RM
from .utils import Beam, build_conv, generate_k_steps, last
from tqdm import tqdm

from collections import defaultdict
logger = logging.getLogger()

# TestTimeComputeAlgorithm is a class that is used to define the interface for all test-time compute algorithms.
class TTCAlgorithm():
    def __init__(self, config:Config, llm:LLM, reward_model:RM):
        self.config = config
        self.llm = llm
        self.reward_model = reward_model
    
    
    def run(batch_of_prompts: list[str]) -> list[Beam]:
        # to run the algorithm on a list of prompts
        raise NotImplementedError
    
    
    def __call__(self, *args, **kwds):
        # to be passed to the datasets map function
        raise NotImplementedError


class BeamSearch(TTCAlgorithm):
    def __init__():
        pass
    
    
    def run(self, batch_of_prompts: list[str]) -> list[Beam]:
        sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=2048,
                top_p=self.config.top_p,
                stop=["\n\n", "\n"],
                include_stop_str_in_output=True,
                n=1,
            )

        beams: list[Beam] = []
        for prompt in batch_of_prompts:
            for i in range(self.config.n_beams):
                beams.append(
                    Beam(
                        prompt=prompt,
                        index=i,
                        current_text="",
                        next_texts=None,
                        lookahead_texts=None,
                        best_scores=[0.0],
                        all_scores=[],
                        previous_text=None,
                        pruned=False,
                        stop_reasons=None,
                        history=[],
                    )
                )

        for i in tqdm(range(self.config.num_iterations), desc="Beam search iterations"):
            # generation
            gen_beams = [b for b in beams if not b.pruned]
            if len(gen_beams) == 0:
                break

            if i == self.config.num_iterations - 1:
                # last iteration, generate to EOS
                sampling_params = SamplingParams(
                    temperature=self.config.temperature,
                    max_tokens=2048,
                    top_p=self.config.top_p,
                    n=1,
                )

            convs = [
                build_conv(b.prompt, b.current_text, self.config.system_prompt)
                for b in gen_beams
            ]
            continue_final_message = i > 0
            add_generation_prompt = i == 0

            tokenizer = self.llm.get_tokenizer()
            # TODO: set the augmented template from a file
            tokenizer.chat_template = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )
            lookahead = 0 if i == self.config.num_iterations - 1 else self.config.lookahead
            gen_results = generate_k_steps(
                templated_convs, lookahead, self.llm, sampling_params, self.config.beam_width
            )

            prompts, completions = [], []
            for beam, gen_result in zip(gen_beams, gen_results, strict=True):
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                if len(beam.next_texts) != self.config.beam_width:
                    beam.pruned = True
                    # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                    logger.warning(
                        f"beam {beam.index} has {len(beam.next_texts)} completions"
                    )
                prompts.append(beam.prompt)
                completions.append([beam.current_text + t for t in beam.lookahead_texts])

            # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt

            all_scores = self.prm.score(prompts, completions)

            agg_fn = last  # TODO: we should strip out the reward for previous steps in the score method
            for beam, scores in zip(gen_beams, all_scores, strict=True):
                agg_scores = [agg_fn(s) for s in scores]
                best_score_ind = np.argmax(agg_scores)
                beam.all_scores = scores
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.history.append(beam.next_texts[best_score_ind])
                beam.best_scores = scores[best_score_ind]
                if (
                    beam.next_texts[best_score_ind] == ""
                    or beam.stop_reasons[best_score_ind] == "EOS"
                ):
                    # stopped on EOS, prune
                    beam.pruned = True

            # filter / prune
            for beam in gen_beams:
                if "boxed{" in beam.current_text:
                    beam.pruned = True

        # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
        output: list[Beam] = []
        for beam in beams:
            for i in range(self.config.beam_width):
                output.append(
                    Beam(
                        prompt=beam.prompt,
                        index=beam.index,
                        current_text=beam.previous_text + beam.next_texts[i],
                        next_texts=None,
                        lookahead_texts=None,
                        stop_reasons=None,
                        best_scores=beam.all_scores[i],
                        all_scores=beam.all_scores,
                        previous_text=beam.current_text,
                        pruned=beam.pruned,
                        history=beam.history,
                    )
                )

        return output

    
    def __call__(self, examples: dict[str, list[str|Any]]) -> dict[str, list[str|Any]]:
        problems = examples["problem"]
        beam_results = self.run(problems)

        # group together alike beams and store in the dataset
        grouped_results = defaultdict(list)
        for results in beam_results:
            grouped_results[results.prompt].append(results)

        results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}
        agg_fn = last  # TODO: we should strip out the reward for previous steps in the score method

        for p in problems:
            beams = grouped_results[p]
            results["completions"].append([b.current_text for b in beams])
            results["pred"].append(
                beams[np.argmax([agg_fn(b.best_scores) for b in beams])].current_text
            )
            results["scores"].append([b.best_scores for b in beams])
            results["completion_tokens"].append(-1)

        # TODO: construct and store the tree

        return results