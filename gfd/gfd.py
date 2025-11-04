from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from gfd.beam import BeamsControler
from gfd.model import BreezeByte
from gfd.tokenizer import LlamaByteTokenizer, TrOCRByteTokenizer

DEBUG = 1


class Breezper:
    def __init__(self, config):
        self.config = config

        self.ocr_processor = TrOCRProcessor.from_pretrained(self.config.ocr_model_path)
        dtype = self._resolve_dtype(getattr(self.config, "ocr_torch_dtype", None))
        if dtype is not None:
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
                self.config.ocr_model_path,
                torch_dtype=dtype,
            )
        else:
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.ocr_model_path)

        self.device = torch.device(self.config.ocr_device)
        self.ocr_model.to(self.device)
        self.ocr_model.eval()

        self.ocr_tokenizer = TrOCRByteTokenizer.from_pretrained(self.config.ocr_model_path)
        self.ocr_model.config.decoder_start_token_id = self.ocr_tokenizer.bos_token_id
        self.ocr_model.config.pad_token_id = self.ocr_tokenizer.pad_token_id
        self.ocr_model.config.eos_token_id = self.ocr_tokenizer.eos_token_id

        self.llm_tokenizer = LlamaByteTokenizer.from_pretrained(self.config.llm_model_path)
        self.breeze_byte = BreezeByte(self.config)

        self.ocr_prefix_template = "{prompt}"
        self.llm_prefix_template = "<s>{prompt}"

    def _resolve_dtype(self, dtype_name):
        if dtype_name is None:
            return None
        if isinstance(dtype_name, str):
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unsupported torch dtype: {dtype_name}")
            return getattr(torch, dtype_name)
        return dtype_name

    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        if isinstance(image_input, str):
            image = Image.open(image_input)
            return image.convert("RGB")
        raise TypeError("Unsupported image input type. Provide a path, numpy array, or PIL.Image.")

    def get_transcription(self, image_input, num_beams=5, ocr_prompt="", llm_prompt=""):
        image = self._load_image(image_input)

        if DEBUG:
            print("ocr prompt:", ocr_prompt)
            print("llm prompt:", llm_prompt)

        pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values.to(self.device)
        encoder_outputs = self.ocr_model.get_encoder()(pixel_values, return_dict=True)

        transcription = self._decode_with_fusion(
            encoder_outputs=encoder_outputs,
            num_beams=num_beams,
            ocr_prompt=ocr_prompt,
            llm_prompt=llm_prompt,
            use_cache=getattr(self.config, "use_cache", None),
        )
        return transcription

    def fuse(self, recognizer_score, llm_score):
        if self.config.fuse_strategy == "simple":
            return (1 - self.config.fusing_r) * recognizer_score + self.config.fusing_r * llm_score
        raise NotImplementedError()

    def _get_prefix_decoding_ids(self, ocr_prompt, llm_prompt):
        ocr_ids = [self.ocr_tokenizer.bos_token_id]
        prompt = (ocr_prompt or "").strip()
        if prompt:
            ocr_ids.extend(self.ocr_tokenizer(prompt, add_special_tokens=False).input_ids)

        llm_prefix_decoding_ids = self.llm_tokenizer.tokenize_from_byte(
            self.llm_prefix_template.format(prompt=llm_prompt).encode("utf8")
        )

        return ocr_ids, llm_prefix_decoding_ids

    def _ocr_forward(self, encoder_outputs, decoder_input_ids, k):
        with torch.no_grad():
            decoder_tensor = torch.tensor([decoder_input_ids], device=self.device)
            logits = self.ocr_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_tensor,
                return_dict=True,
            ).logits
            logprobs = torch.log_softmax(logits, dim=-1)
            next_logprobs, inds = torch.topk(logprobs[0, -1, :], k, dim=-1)

        return next_logprobs, inds

    def _decode_with_fusion(self, encoder_outputs, num_beams, ocr_prompt, llm_prompt, use_cache=None):
        beams = BeamsControler(
            config=self.config,
            n_beam=num_beams,
            asr_eos_id=self.ocr_tokenizer.eos_token_id,
        )

        ocr_prefix_ids, llm_prefix_ids = self._get_prefix_decoding_ids(ocr_prompt, llm_prompt)
        next_scores, next_tokens = self._ocr_forward(encoder_outputs, ocr_prefix_ids, k=1)

        for token_id, token_score in zip(next_tokens, next_scores):
            next_id = token_id.item()
            score = token_score.item()
            recognizer_score, llm_score = self._calcualte_asr_llm_score(
                asr_normalized_len=1,
                asr_logprob=score,
                llm_normalized_len=1,
                llm_logprob=None,
            )
            fuse_score = self.fuse(recognizer_score, llm_score)
            beams.add(
                asr_score=recognizer_score,
                llm_score=llm_score,
                fuse_score=fuse_score,
                asr_prefix_ids=ocr_prefix_ids,
                asr_ids=[next_id],
                asr_logprob=score,
                llm_prefix_ids=llm_prefix_ids,
                llm_ids=[],
                llm_logprob=None,
            )

        self._update_asr_llm_mean_and_std(beams._next_beams)
        beams.update()

        while True:
            for beam in beams.list():
                if beam.reach_end:
                    beams.add_beam(beam)
                    continue

                next_scores, next_tokens = self._ocr_forward(
                    encoder_outputs,
                    beam.asr_prefix_ids + beam.asr_ids,
                    k=num_beams,
                )

                next_tokens = [token.item() for token in next_tokens]
                next_scores = [score.item() for score in next_scores]

                if next_tokens and next_tokens[0] == self.ocr_tokenizer.eos_token_id:
                    next_tokens = next_tokens[:1]
                    next_scores = next_scores[:1]
                elif self.ocr_tokenizer.eos_token_id in next_tokens:
                    eos_idx = next_tokens.index(self.ocr_tokenizer.eos_token_id)
                    next_tokens = next_tokens[:eos_idx] + next_tokens[eos_idx + 1 :]
                    next_scores = next_scores[:eos_idx] + next_scores[eos_idx + 1 :]

                recognizer_bytes = self.ocr_tokenizer.convert_ids_to_bytes(
                    beam.asr_ids,
                    skip_special_tokens=True,
                )
                new_content = b"".join(recognizer_bytes)
                llm_ids = self.llm_tokenizer.tokenize_from_byte(new_content)

                if use_cache == "dynamic":
                    llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob_cache_dynamic(
                        prefix_decoding_ids=llm_prefix_ids,
                        llm_ids=llm_ids,
                        llm_tokenizer=self.llm_tokenizer,
                    )
                elif use_cache == "static":
                    llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob_cache_static(
                        prefix_decoding_ids=llm_prefix_ids,
                        llm_ids=llm_ids,
                        llm_tokenizer=self.llm_tokenizer,
                    )
                else:
                    llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob(
                        prefix_decoding_ids=llm_prefix_ids,
                        llm_ids=llm_ids,
                        llm_tokenizer=self.llm_tokenizer,
                    )

                assert normalizer_adjust_n <= 0

                for next_id, score in zip(next_tokens, next_scores):
                    recognizer_logprob = score + beam.asr_logprob
                    recognizer_score, llm_score = self._calcualte_asr_llm_score(
                        asr_normalized_len=len(beam.asr_ids) + 1,
                        asr_logprob=recognizer_logprob,
                        llm_normalized_len=len(llm_ids) + normalizer_adjust_n,
                        llm_logprob=llm_logprob,
                    )
                    fuse_score = self.fuse(recognizer_score, llm_score)
                    beams.add(
                        asr_score=recognizer_score,
                        llm_score=llm_score,
                        fuse_score=fuse_score,
                        asr_prefix_ids=beam.asr_prefix_ids,
                        asr_ids=beam.asr_ids + [next_id],
                        asr_logprob=recognizer_logprob,
                        llm_prefix_ids=beam.llm_prefix_ids,
                        llm_ids=llm_ids,
                        llm_logprob=llm_logprob,
                    )

            self._update_asr_llm_mean_and_std(beams._next_beams)
            beams.update()
            self.breeze_byte.kv_cache.remove_unused()

            if DEBUG > 1:
                for idx, beam in enumerate(beams.list()):
                    print(
                        f"[{idx}] recognizer_score={beam.asr_score}, "
                        f"llm_score={beam.llm_score}, fuse_score={beam.fuse_score},\n"
                        f"{self.ocr_tokenizer.decode(beam.asr_ids)}"
                    )
                print()
            elif DEBUG > 0 and beams.list():
                beam = beams.list()[0]
                print(
                    f"[0] recognizer_score={beam.asr_score}, "
                    f"llm_score={beam.llm_score}, fuse_score={beam.fuse_score},\n"
                    f"{self.ocr_tokenizer.decode(beam.asr_ids)}\n"
                )

            if beams.is_terminated():
                break

        transcription = beams.get_result(self.ocr_tokenizer)
        return transcription

    def _update_asr_llm_mean_and_std(self, beams_list):
        # DEPRECATED
        pass

    def _calcualte_asr_llm_score(self, asr_normalized_len, asr_logprob, llm_normalized_len, llm_logprob):
        if not (asr_logprob > self.config.logprob_min):
            asr_logprob = self.config.logprob_min

        if llm_logprob is None or not (llm_logprob > self.config.logprob_min):
            llm_logprob = self.config.logprob_min

        asr_score = asr_logprob / asr_normalized_len if asr_normalized_len > 0 else self.config.logprob_min
        llm_score = llm_logprob / llm_normalized_len if llm_normalized_len > 0 else self.config.logprob_min

        return asr_score, llm_score
