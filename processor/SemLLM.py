import torch
import pandas as pd
from typing import Iterable, List
from tqdm.auto import tqdm
from typing import Optional, Union, cast
import itertools
from vllm import LLM
from vllm.beam_search import (BeamSearchInstance, BeamSearchOutput,
                              BeamSearchSequence,
                              create_sort_beams_key_function)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import (BeamSearchParams, SamplingParams)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, BatchUpdate
from vllm.config import VllmConfig
from typing import List, Tuple, Set, Optional

import torch
import pandas as pd
from typing import Optional
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, BatchUpdate
from vllm.config import VllmConfig


class SemLLM(LLM):
    def beam_search(
        self,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: BeamSearchParams,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        use_tqdm: bool = False,
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        lora_requests = self._get_beam_search_lora_requests(
            lora_request, prompts)

        tokenizer = self.get_tokenizer()
        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id,
            length_penalty,
        )

        def create_tokens_prompt_from_beam(
                beam: BeamSearchSequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {
                "prompt_token_ids": beam.tokens
            }
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs[
                    "mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, prompts):
            # Add multimodal processor kwargs & data
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt[
                    "mm_processor_kwargs"]

            if "prompt_token_ids" in prompt:
                prompt = cast(TokensPrompt, prompt)  # Needed for mypy
                prompt_tokens = prompt["prompt_token_ids"]
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"])

            instances.append(
                BeamSearchInstance(
                    prompt_tokens,
                    lora_request=lora_req,
                    logprobs=None,
                    **mm_kwargs,
                ), )

        token_iter = range(max_tokens)
        if use_tqdm:
            token_iter = tqdm(token_iter,
                                desc="Beam search",
                                unit="token",
                                unit_scale=False)

        for idx in token_iter:
            all_beams: list[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: list[tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            # create the corresponding batch entries for prompt & optional lora
            prompts_batch, lora_req_batch = zip(
                *[(create_tokens_prompt_from_beam(beam), beam.lora_request)
                    for beam in all_beams])

            output = self.generate(prompts_batch,
                                    sampling_params=beam_search_params,
                                    use_tqdm=False,
                                    lora_request=lora_req_batch)

            for (start, end), instance in zip(instance_start_and_end,
                                                instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                logprobs=current_beam.logprobs + [logprobs],
                                lora_request=current_beam.lora_request,
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob,
                                multi_modal_data=current_beam.multi_modal_data,
                                mm_processor_kwargs=current_beam.
                                mm_processor_kwargs)

                            if token_id == tokenizer.eos_token_id and \
                                not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                        key=sort_beams_key,
                                        reverse=True)
                instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                        key=sort_beams_key,
                                        reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs