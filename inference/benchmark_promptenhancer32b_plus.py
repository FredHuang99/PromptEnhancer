import os
import sys
import time
import json
import math
import argparse
from threading import Thread
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer

# Ensure project root is on PYTHONPATH when running this file directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.prompt_enhancer_v2 import PromptEnhancerV2


# 为了和你上一版 benchmark 风格保持一致，这里保留一个 sys prompt。
# 如果你想全英文模板，可以改成：
# 请根据用户的输入，生成思考过程的思维链并改写提示词：
DEFAULT_SYS_PROMPT = "Please think step by step and rewrite the user's prompt for text-to-image generation while preserving the original intent."


class TimingStreamer(BaseStreamer):
    """
    Records per-token timestamps.
    Assumes batch size = 1.
    """
    def __init__(self):
        super().__init__()
        self.next_tokens_are_prompt = True
        self.token_times = []
        self.token_ids = []
        self.ended = False
        self.end_time = None

    def put(self, value):
        if torch.is_tensor(value):
            if value.dim() > 1:
                value = value[0]
            ids = value.tolist()
        else:
            ids = list(value)

        # skip prompt tokens
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        now = time.perf_counter()
        for tid in ids:
            self.token_times.append(now)
            self.token_ids.append(int(tid))

    def end(self):
        self.ended = True
        self.end_time = time.perf_counter()


def sync_all_visible_gpus():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def normalize_prompt(s: str) -> str:
    # prompt.txt 要求一行一个 prompt，所以把内部换行压成空格
    return " ".join(str(s).strip().split())


def load_en_prompts_from_hf(limit: Optional[int] = None,
                            file_path: str = None) -> List[str]:
    if file_path:
        ds = load_dataset(file_path, split="train")
    else:
        ds = load_dataset("PromptEnhancer/T2I-Keypoints-Eval", split="train")
    prompts = []
    for item in ds:
        if item.get("language") == "en" and item.get("prompt"):
            prompts.append(normalize_prompt(item["prompt"]))
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def load_prompts_from_txt(path: str, limit: Optional[int] = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = [normalize_prompt(line) for line in f if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def save_text_lines(lines: List[str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(f"{x}\n")


def save_numeric_lines(values: List[float], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for x in values:
            f.write(f"{x}\n")


def get_input_token_length(tokenizer, user_prompt: str, sys_prompt: str) -> int:
    ids = tokenizer(user_prompt, return_tensors="pt")["input_ids"]
    return int(ids.shape[1])


def run_once(
    enhancer,
    prompt: str,
    sys_prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 5,
    use_cache: bool = True,
):
    inputs = enhancer.build_inputs(prompt, sys_prompt, device="cuda")
    streamer = TimingStreamer()

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        use_cache=use_cache,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    sync_all_visible_gpus()
    t0 = time.perf_counter()

    err_holder = []

    def _generate_with_inference_mode():
        try:
            with torch.inference_mode():
                enhancer.model.generate(**gen_kwargs)
        except Exception as e:
            err_holder.append(e)

    th = Thread(target=_generate_with_inference_mode)
    th.start()
    th.join()

    if err_holder:
        raise err_holder[0]

    sync_all_visible_gpus()
    t1 = time.perf_counter()

    num_out_tokens = len(streamer.token_ids)
    if num_out_tokens == 0:
        return {
            "prompt": prompt,
            "num_out_tokens": -1,
            "ttft_ms": -1.0,
            "tpot_ms": -1.0,
            "e2e_ms": (t1 - t0) * 1000.0,
        }

    ttft = streamer.token_times[0] - t0
    if num_out_tokens >= 2:
        inter = np.diff(streamer.token_times)
        tpot = float(np.mean(inter))
    else:
        tpot = float("nan")

    return {
        "prompt": prompt,
        "num_out_tokens": int(num_out_tokens),
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": -1.0 if math.isnan(tpot) else tpot * 1000.0,
        "e2e_ms": (t1 - t0) * 1000.0,
    }


def summarize_numeric(vals: List[float]):
    vals = [x for x in vals if x >= 0]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()

    # 数据来源
    parser.add_argument("--use-hf-en", action="store_true",
                        help="Use English subset of PromptEnhancer/T2I-Keypoints-Eval")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Local txt file, one prompt per line")
    parser.add_argument("--prompt-dataset", type=str, default=None,
                        help="Local prompt dataset in HuggingFace format, e.g. ./my_prompts")
    parser.add_argument("--limit", type=int, default=None)

    # 模型 / processor
    parser.add_argument("--model", type=str, default=None,
                        help="Local model path, e.g. ./models/promptenhancer-32b")
    parser.add_argument("--sys-prompt", type=str, default=DEFAULT_SYS_PROMPT)

    # 导出功能
    parser.add_argument("--export-prompts-txt", type=str, default=None,
                        help="Export prompts to txt, one prompt per line")
    parser.add_argument("--export-input-lens-txt", type=str, default=None,
                        help="Export input token lengths to txt, one number per line")
    parser.add_argument("--export-output-lens-txt", type=str, default=None,
                        help="Export output token lengths to txt, one number per line")
    parser.add_argument("--benchmark-e2e-txt", type=str, default=None,
                        help="Export per-request e2e latency (ms) to txt, one number per line")
    parser.add_argument("--benchmark-json", type=str, default=None,
                        help="Optional JSON dump for benchmark results")

    # 推理参数
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable KV cache during generation (default: true)")

    args = parser.parse_args()

    if not args.use_hf_en and not args.prompt_file:
        raise ValueError("Please specify either --use-hf-en or --prompt-file")

    # 先拿 prompts
    if args.use_hf_en:
        prompts = load_en_prompts_from_hf(limit=args.limit, file_path=args.prompt_dataset)
        print(f"Loaded {len(prompts)} English prompts from HF dataset.")
    else:
        prompts = load_prompts_from_txt(args.prompt_file, limit=args.limit)
        print(f"Loaded {len(prompts)} prompts from local txt.")

    # 功能 4：导出 prompt.txt
    if args.export_prompts_txt is not None:
        save_text_lines(prompts, args.export_prompts_txt)
        print(f"[OK] Saved prompts txt to: {args.export_prompts_txt}")

    need_processor = args.export_input_lens_txt is not None
    need_model = (
        args.export_output_lens_txt is not None
        or args.benchmark_e2e_txt is not None
        or args.benchmark_json is not None
    )

    if (need_processor or need_model) and not args.model:
        raise ValueError("--model is required for token-length / inference related functions")

    # 功能 2：统计英文数据输入长度
    if need_processor:
        print("Loading tokenizer only...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        input_lens = []
        for i, p in enumerate(prompts):
            try:
                n = get_input_token_length(tokenizer, p, args.sys_prompt)
            except Exception as e:
                print(f"[WARN] input length failed at idx={i}: {e}")
                n = -1
            input_lens.append(int(n))
            if (i + 1) % 100 == 0 or (i + 1) == len(prompts):
                print(f"[input_len] {i+1}/{len(prompts)}")
        save_numeric_lines(input_lens, args.export_input_lens_txt)
        print(f"[OK] Saved input token lengths to: {args.export_input_lens_txt}")
        print("[Summary][input_tokens]", json.dumps(summarize_numeric(input_lens), indent=2, ensure_ascii=False))

    # 功能 1 + 3：e2e / 输出 token 长度
    if need_model:
        do_sample = args.temperature > 0.0

        print("Loading PromptEnhancerV2 model...")
        enhancer = PromptEnhancerV2(
            models_root_path=args.model,
            device_map="auto",
        )

        if hasattr(enhancer.model, "hf_device_map"):
            print("hf_device_map =", enhancer.model.hf_device_map)

        warmup_n = min(args.warmup, len(prompts))
        print(f"Warmup x {warmup_n}")
        for i in range(warmup_n):
            _ = run_once(
                enhancer,
                prompts[i],
                sys_prompt=args.sys_prompt,
                max_new_tokens=min(64, args.max_new_tokens),
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=args.use_cache,
            )

        results = []
        for i, p in enumerate(prompts):
            try:
                r = run_once(
                    enhancer,
                    p,
                    sys_prompt=args.sys_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    use_cache=args.use_cache,
                )
            except Exception as e:
                print(f"[WARN] generation failed at idx={i}: {e}")
                r = {
                    "prompt": p,
                    "num_out_tokens": -1,
                    "ttft_ms": -1.0,
                    "tpot_ms": -1.0,
                    "e2e_ms": -1.0,
                }

            results.append(r)
            print(
                f"[{i+1}/{len(prompts)}] "
                f"out_tokens={r['num_out_tokens']}, "
                f"ttft_ms={r['ttft_ms']:.3f}, "
                f"tpot_ms={r['tpot_ms']:.3f}, "
                f"e2e_ms={r['e2e_ms']:.3f}"
            )

        if args.benchmark_e2e_txt is not None:
            e2e_vals = [x["e2e_ms"] for x in results]
            save_numeric_lines(e2e_vals, args.benchmark_e2e_txt)
            print(f"[OK] Saved e2e latencies to: {args.benchmark_e2e_txt}")
            print("[Summary][e2e_ms]", json.dumps(summarize_numeric(e2e_vals), indent=2, ensure_ascii=False))

        if args.export_output_lens_txt is not None:
            out_lens = [x["num_out_tokens"] for x in results]
            save_numeric_lines(out_lens, args.export_output_lens_txt)
            print(f"[OK] Saved output token lengths to: {args.export_output_lens_txt}")
            print("[Summary][output_tokens]", json.dumps(summarize_numeric(out_lens), indent=2, ensure_ascii=False))

        if args.benchmark_json is not None:
            payload = {
                "args": vars(args),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "summary": {
                    "e2e_ms": summarize_numeric([x["e2e_ms"] for x in results]),
                    "ttft_ms": summarize_numeric([x["ttft_ms"] for x in results]),
                    "tpot_ms": summarize_numeric([x["tpot_ms"] for x in results]),
                    "output_tokens": summarize_numeric([x["num_out_tokens"] for x in results]),
                },
                "results": results,
            }
            with open(args.benchmark_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved benchmark json to: {args.benchmark_json}")


if __name__ == "__main__":
    main()
