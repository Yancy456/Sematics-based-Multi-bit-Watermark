# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
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

from __future__ import annotations
import collections
import math

import random

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup


class WatermarkBase:
    def __init__(
        self,
        num_colors: int,  # number of colors in colorlist
        message: str = None,  # the message to be embedded. Message should be converted into string that only contains 0 or 1.
        message_len: int = None,  # length of original message (only used when detection)
        vocab: list[int] = None,
        # gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        # select_green_tokens: bool = True
    ):
        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.message = message
        self.message_len = message_len if message_len is not None else len(message)  # message_len is only used in detection. And message is only in embedding.
        self.num_colors = num_colors
        self.converted_message = self._radix_convert(message, 2, num_colors, self.message_len) if message is not None else None  # convert binary string into r radix string
        self.b_hat = int(math.ceil(self.message_len / math.log(self.num_colors, 2)))

    def _radix_convert(self, message: str, input_r: int, output_r: int, message_len: int) -> str:
        '''convert input_r radix input string into output_r radix output string'''
        # Convert binary string to an integer
        try:
            num = int(message, input_r)
        except ValueError:
            print('Embedded message must be binary string')
            return

        digits = []
        while num:
            digits.append(int(num % output_r))
            num //= output_r
        return ''.join(str(x) for x in digits[::-1])

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_p_and_color_idx(self, previous_ids: torch.LongTensor, current_id: torch.LongTensor):
        '''Return the position and color_idx for current token
        '''
        self._seed_rng(previous_ids)
        color_size = int(self.vocab_size / self.num_colors)  # size of one color
        vocab_permutation = torch.randperm(self.vocab_size, device=previous_ids.device, generator=self.rng)
        position = torch.randint(0, self.b_hat, size=(1,), device=current_id.device, generator=self.rng).item()

        color_idx = None  # if color_idx is None, it means current_id isn't in any color strip

        for i in range(self.num_colors):
            if current_id.item() in vocab_permutation[i * color_size:(i + 1) * color_size].tolist():
                color_idx = i
                break

        return position, color_idx

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        '''input_ids are the previous tokens used to seed genarator
        Return the selected color from colorlist as greenlist(only used in watermark embedding)
        '''
        self._seed_rng(input_ids)

        color_size = int(self.vocab_size / self.num_colors)  # size of one color

        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)

        # this is colorlist
        # colorlist_ids=[vocab_permutation[i*color_size:(i+1)*color_size] for i in range(self.num_colors)]
        b_hat = self.b_hat

        position = torch.randint(0, b_hat, size=(1,), device=input_ids.device, generator=self.rng).item()

        m = int(self.converted_message[b_hat - position - 1])

        greenlist_ids = vocab_permutation[m * color_size:(m + 1) * color_size]

        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, num_colors, message, **kwargs):
        super().__init__(num_colors=num_colors, message=message, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        # z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        # self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    # def _compute_z_score(self, observed_count, T):
    #    # count refers to number of green tokens, T is total number of tokens
    #    expected_count = self.gamma
    #    numer = observed_count - expected_count * T
    #    denom = sqrt(T * expected_count * (1 - expected_count))
    #    z = numer / denom
    #    return z

    # def _compute_p_value(self, z):
    #    p_value = scipy.stats.norm.sf(z)
    #    return p_value

    def _decode_sequence(
        self,
        input_ids: Tensor
    ):
        if self.ignore_repeated_bigrams:
            '''Only counts a green/red hit for unique bigram, not each word. This would add some robustness.
            '''
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []

            b_hat = self.b_hat    # maximun length of converted message

            w = torch.zeros(size=(b_hat, self.num_colors), dtype=torch.int8)  # counter for message decoding

            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                p, color_idx = self._get_p_and_color_idx(input_ids[:idx], curr_token)

                if color_idx is not None:
                    w[p, color_idx] += 1

            converted_message = torch.argmax(w, dim=1)
            print(converted_message)
            converted_message = converted_message.tolist()[::-1]
            converted_message = ''.join([str(x) for x in converted_message])
            message = self._radix_convert(converted_message, self.num_colors, 2, self.message_len)

        return {'message': message}

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            '''Only counts a green/red hit for unique bigram, not each word. This would add some robustness.
            '''
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx]
                greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        # return_prediction: bool = True,
        # return_scores: bool = True,
        # z_threshold: float = None,
        **kwargs,
    ) -> dict:
        '''Return the decode result of text
        '''

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}
        score_dict = self._decode_sequence(tokenized_text, **kwargs)
        output_dict.update(score_dict)

        # if return_scores:
        #    output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        # if return_prediction:
        #    z_threshold = z_threshold if z_threshold else self.z_threshold
        #    assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
        #    output_dict["prediction"] = score_dict["z_score"] > z_threshold
        #    if output_dict["prediction"]:
        #        output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
