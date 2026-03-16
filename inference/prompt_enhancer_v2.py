"""
Copyright (c) 2025 Tencent. All Rights Reserved.
Licensed under the Tencent Hunyuan Community License Agreement.
"""

import importlib.util
import logging
import os
import re
import time
from typing import Any, Dict, List

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

DEFAULT_SYS_PROMPT = (
    "Please think step by step and rewrite the user's prompt for text-to-image generation "
    "while preserving the original intent."
)


def replace_single_quotes(text: str) -> str:
    """
    Replace single quotes within words with double quotes.
    """
    pattern = r"\B'([^']*)'\B"
    return re.sub(pattern, r'"\1"', text)


class PromptEnhancerV2:
    def __init__(self, models_root_path: str, device_map: str = "auto"):
        """
        Initialize model and processor/tokenizer with automatic backend selection.

        Supports both:
        - qwen2_5_vl checkpoints (vision-language)
        - hunyuan_v1_dense checkpoints (text-only CausalLM)
        """
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.models_root_path = models_root_path
        self.device_map = device_map
        self.config = AutoConfig.from_pretrained(models_root_path, trust_remote_code=True)
        self.model_type = str(getattr(self.config, "model_type", ""))
        self.is_vl = self.model_type == "qwen2_5_vl"
        self.is_hunyuan_dense = self.model_type == "hunyuan_v1_dense"

        dtype = self._pick_dtype()
        attn_impl = self._preferred_attn_impl()

        common_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }

        if self.is_vl:
            model_cls = Qwen2_5_VLForConditionalGeneration
            model_kwargs = dict(common_kwargs)
            # Qwen2.5-VL does not need trust_remote_code here.
        elif self.is_hunyuan_dense:
            model_cls = AutoModelForCausalLM
            model_kwargs = dict(common_kwargs)
            model_kwargs["trust_remote_code"] = True
        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                "Expected one of: qwen2_5_vl, hunyuan_v1_dense."
            )

        self.model, self.attn_implementation = self._load_model_with_fallback(
            model_cls=model_cls,
            model_path=models_root_path,
            base_kwargs=model_kwargs,
            preferred_attn_impl=attn_impl,
        )

        # Keep the attribute name `processor` so existing call sites continue to work.
        if self.is_vl:
            self.processor = AutoProcessor.from_pretrained(models_root_path)
        else:
            self.processor = AutoTokenizer.from_pretrained(
                models_root_path, trust_remote_code=True
            )

        self.logger.info(
            "Loaded model_type=%s, backend=%s, attn_implementation=%s",
            self.model_type,
            "Qwen2.5-VL" if self.is_vl else "HunyuanDense CausalLM",
            self.attn_implementation,
        )

    def _pick_dtype(self):
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _preferred_attn_impl(self) -> str:
        # Prefer FA2 only when CUDA + flash_attn are both available.
        if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    def _load_model_with_fallback(
        self,
        model_cls,
        model_path: str,
        base_kwargs: Dict[str, Any],
        preferred_attn_impl: str,
    ):
        kwargs = dict(base_kwargs)
        kwargs["attn_implementation"] = preferred_attn_impl

        try:
            return model_cls.from_pretrained(model_path, **kwargs), preferred_attn_impl
        except ImportError as e:
            if preferred_attn_impl == "flash_attention_2" and "flash_attn" in str(e):
                self.logger.warning(
                    "flash_attn is unavailable; fallback to sdpa attention. Original error: %s", e
                )
                kwargs["attn_implementation"] = "sdpa"
                return model_cls.from_pretrained(model_path, **kwargs), "sdpa"
            raise
        except ValueError as e:
            # Some backends may reject specific attention names.
            if "attn_implementation" in str(e):
                kwargs.pop("attn_implementation", None)
                return model_cls.from_pretrained(model_path, **kwargs), "default"
            raise
        except TypeError:
            # Some custom models may not accept attn_implementation.
            kwargs.pop("attn_implementation", None)
            return model_cls.from_pretrained(model_path, **kwargs), "default"

    def build_inputs(self, user_prompt: str, sys_prompt: str, device: str = "cuda"):
        """
        Build generation inputs for both VL and text-only backends.
        """
        if self.is_vl:
            from qwen_vl_utils import process_vision_info

            merged = f"{sys_prompt}\n{user_prompt}" if sys_prompt else user_prompt
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": merged}],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            messages: List[Dict[str, str]] = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})

            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = f"{sys_prompt}\n{user_prompt}" if sys_prompt else user_prompt

            inputs = self.processor(
                [text],
                padding=True,
                return_tensors="pt",
            )

        return inputs.to(device)

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot: str,
        sys_prompt: str = DEFAULT_SYS_PROMPT,
        temperature: float = 0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        device: str = "cuda",
    ) -> str:
        """
        Generate a rewritten prompt; fallback to original prompt on failure.
        """
        org_prompt_cot = prompt_cot
        try:
            inputs = self.build_inputs(prompt_cot, sys_prompt, device=device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                do_sample=bool(temperature > 0),
                top_k=5,
                top_p=float(top_p),
                use_cache=True,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_res = output_text[0]

            # Keep compatibility with the old output parser.
            if output_res.count("think>") >= 2:
                prompt_cot = output_res.split("think>")[-1]
            else:
                prompt_cot = output_res

            if prompt_cot.startswith("\n"):
                prompt_cot = prompt_cot[1:]
            prompt_cot = replace_single_quotes(prompt_cot)
        except Exception:
            prompt_cot = org_prompt_cot
            print("Re-prompting failed, so using the original prompt")

        return prompt_cot


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_OUTPUT_PATH", "/path/to/your/model")
    prompt_enhancer_cls = PromptEnhancerV2(models_root_path=model_path)

    test_list_en = [
        "Create a painting depicting a 30-year-old white female white-collar worker on a business trip by plane.",
        "Depicted in the anime style of Studio Ghibli, a girl stands quietly at the deck with a gentle smile.",
        "Blue background, a lone girl gazes into the distant sea; her expression is sorrowful.",
    ]

    print("Testing prompts:")
    for item in test_list_en:
        print("User Prompt:", item)
        time_start = time.time()
        result = prompt_enhancer_cls.predict(item)
        time_end = time.time()
        print("RePrompt:", result)
        print("Time cost:", time_end - time_start)
        print("~~~~~~~~~~~~~~")
