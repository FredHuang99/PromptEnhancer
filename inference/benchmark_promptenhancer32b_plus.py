import os
import sys
import time
import json
import math
import argparse
import builtins
from threading import Thread
from functools import partial
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

# Ensure real-time logs in nohup/non-interactive runs.
print = partial(builtins.print, flush=True)


# 娑撹桨绨￠崪灞肩稑娑撳﹣绔撮悧?benchmark 妞嬪孩鐗告穱婵囧瘮娑撯偓閼疯揪绱濇潻娆撳櫡娣囨繄鏆€娑撯偓娑?sys prompt閵?
# 婵″倹鐏夋担鐘冲厒閸忋劏瀚抽弬鍥侀弶鍖＄礉閸欘垯浜掗弨瑙勫灇閿?
# 鐠囬攱鐗撮幑顔炬暏閹撮娈戞潏鎾冲弳閿涘瞼鏁撻幋鎰偓婵娾偓鍐箖缁嬪娈戦幀婵堟樊闁炬儳鑻熼弨鐟板晸閹绘劗銇氱拠宥忕窗
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
    # prompt.txt 鐟曚焦鐪版稉鈧悰灞肩娑?prompt閿涘本澧嶆禒銉﹀Ω閸愬懘鍎撮幑銏ｎ攽閸樺鍨氱粚鐑樼壐
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


def append_numeric_lines(values: List[float], out_path: str):
    if not values:
        return
    with open(out_path, "a", encoding="utf-8") as f:
        for x in values:
            f.write(f"{x}\n")


def load_numeric_lines(path: str) -> List[float]:
    if not path or not os.path.exists(path):
        return []
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                continue
    return vals


def count_non_empty_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


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
        decode_end = streamer.end_time if streamer.end_time is not None else t1
        tpot = (decode_end - streamer.token_times[0]) / float(num_out_tokens - 1)
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

    # 閺佺増宓侀弶銉︾爱
    parser.add_argument("--use-hf-en", action="store_true",
                        help="Use English subset of PromptEnhancer/T2I-Keypoints-Eval")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Local txt file, one prompt per line")
    parser.add_argument("--prompt-dataset", type=str, default=None,
                        help="Local prompt dataset in HuggingFace format, e.g. ./my_prompts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index (inclusive) after prompt loading/filtering")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index (exclusive) after prompt loading/filtering")

    # 濡€崇€?/ processor
    parser.add_argument("--model", type=str, default=None,
                        help="Local model path, e.g. ./models/promptenhancer-32b")
    parser.add_argument("--sys-prompt", type=str, default=DEFAULT_SYS_PROMPT)

    # 鐎电厧鍤崝鐔诲厴
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

    # 閹恒劎鎮婇崣鍌涙殶
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable KV cache during generation (default: true)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Incremental flush interval for e2e/output-lens txt (default: 50)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from existing e2e/output-lens txt files (default: true)")

    args = parser.parse_args()

    if not args.use_hf_en and not args.prompt_file:
        raise ValueError("Please specify either --use-hf-en or --prompt-file")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < 0:
        raise ValueError("--end-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    # 閸忓牊瀣?prompts
    if args.use_hf_en:
        prompts = load_en_prompts_from_hf(limit=None, file_path=args.prompt_dataset)
        print(f"Loaded {len(prompts)} English prompts from HF dataset (before slicing).")
    else:
        prompts = load_prompts_from_txt(args.prompt_file, limit=None)
        print(f"Loaded {len(prompts)} prompts from local txt (before slicing).")

    total_before_slice = len(prompts)
    start_idx = min(args.start_idx, total_before_slice)
    end_idx = total_before_slice if args.end_idx is None else min(args.end_idx, total_before_slice)
    prompts = prompts[start_idx:end_idx]
    if args.limit is not None:
        prompts = prompts[:args.limit]
    print(
        f"Using prompt range [{start_idx}, {end_idx}) "
        f"with limit={args.limit}, final_count={len(prompts)}"
    )
    if len(prompts) == 0:
        raise ValueError("No prompts selected after applying range/limit settings")

    # 閸旂喕鍏?4閿涙艾顕遍崙?prompt.txt
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

    # 閸旂喕鍏?2閿涙氨绮虹拋陇瀚抽弬鍥ㄦ殶閹诡喛绶崗銉╂毐鎼?
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

    # 閸旂喕鍏?1 + 3閿涙瓱2e / 鏉堟挸鍤?token 闂€鍨
    if need_model:
        do_sample = args.temperature > 0.0

        # Resume is based on existing line counts in output txt files.
        resume_count = 0
        resume_counts = []
        if args.resume:
            if args.benchmark_e2e_txt is not None:
                resume_counts.append(count_non_empty_lines(args.benchmark_e2e_txt))
            if args.export_output_lens_txt is not None:
                resume_counts.append(count_non_empty_lines(args.export_output_lens_txt))
            if resume_counts:
                resume_count = min(resume_counts)
                if len(set(resume_counts)) > 1:
                    print(
                        "[WARN] Existing txt line counts are inconsistent: "
                        f"{resume_counts}. Resume will use min={resume_count}."
                    )
        else:
            if args.benchmark_e2e_txt is not None:
                open(args.benchmark_e2e_txt, "w", encoding="utf-8").close()
            if args.export_output_lens_txt is not None:
                open(args.export_output_lens_txt, "w", encoding="utf-8").close()

        if resume_count > 0:
            print(f"[RESUME] Skip first {resume_count} prompts in selected range.")
        if resume_count >= len(prompts):
            print("[RESUME] Selected range is already fully processed.")
            if args.benchmark_e2e_txt is not None:
                e2e_vals = load_numeric_lines(args.benchmark_e2e_txt)
                print("[Summary][e2e_ms]", json.dumps(summarize_numeric(e2e_vals), indent=2, ensure_ascii=False))
            if args.export_output_lens_txt is not None:
                out_lens = load_numeric_lines(args.export_output_lens_txt)
                print("[Summary][output_tokens]", json.dumps(summarize_numeric(out_lens), indent=2, ensure_ascii=False))
            if args.benchmark_json is not None:
                payload = {
                    "args": vars(args),
                    "selected_range": {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "final_count": len(prompts),
                        "resume_count": resume_count,
                    },
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "summary": {
                        "e2e_ms": summarize_numeric(load_numeric_lines(args.benchmark_e2e_txt)) if args.benchmark_e2e_txt else {},
                        "output_tokens": summarize_numeric(load_numeric_lines(args.export_output_lens_txt)) if args.export_output_lens_txt else {},
                    },
                    "results": [],
                }
                with open(args.benchmark_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"[OK] Saved benchmark json to: {args.benchmark_json}")
            return

        remaining_prompts = prompts[resume_count:]

        print("Loading PromptEnhancerV2 model...")
        enhancer = PromptEnhancerV2(
            models_root_path=args.model,
            device_map="auto",
        )

        if hasattr(enhancer.model, "hf_device_map"):
            print("hf_device_map =", enhancer.model.hf_device_map)

        if hasattr(enhancer.model, "config"):
            enhancer.model.config.use_cache = bool(args.use_cache)
        if hasattr(enhancer.model, "generation_config"):
            enhancer.model.generation_config.use_cache = bool(args.use_cache)
        print(
            f"[CACHE] requested={bool(args.use_cache)} "
            f"model.config.use_cache={getattr(getattr(enhancer.model, 'config', None), 'use_cache', None)} "
            f"generation_config.use_cache={getattr(getattr(enhancer.model, 'generation_config', None), 'use_cache', None)}"
        )

        warmup_n = min(args.warmup, len(remaining_prompts))
        print(f"Warmup x {warmup_n}")
        for i in range(warmup_n):
            _ = run_once(
                enhancer,
                remaining_prompts[i],
                sys_prompt=args.sys_prompt,
                max_new_tokens=min(64, args.max_new_tokens),
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=args.use_cache,
            )

        results = []
        e2e_buffer: List[float] = []
        out_len_buffer: List[float] = []

        def flush_incremental_buffers():
            if args.benchmark_e2e_txt is not None and e2e_buffer:
                append_numeric_lines(e2e_buffer, args.benchmark_e2e_txt)
                e2e_buffer.clear()
            if args.export_output_lens_txt is not None and out_len_buffer:
                append_numeric_lines(out_len_buffer, args.export_output_lens_txt)
                out_len_buffer.clear()

        for local_i, p in enumerate(remaining_prompts):
            global_i = resume_count + local_i
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
                print(f"[WARN] generation failed at idx={global_i}: {e}")
                r = {
                    "prompt": p,
                    "num_out_tokens": -1,
                    "ttft_ms": -1.0,
                    "tpot_ms": -1.0,
                    "e2e_ms": -1.0,
                }

            results.append(r)
            if args.benchmark_e2e_txt is not None:
                e2e_buffer.append(float(r["e2e_ms"]))
            if args.export_output_lens_txt is not None:
                out_len_buffer.append(float(r["num_out_tokens"]))

            if ((local_i + 1) % args.save_every) == 0:
                flush_incremental_buffers()

            print(
                f"[{global_i+1}/{len(prompts)}] "
                f"out_tokens={r['num_out_tokens']}, "
                f"ttft_ms={r['ttft_ms']:.3f}, "
                f"tpot_ms={r['tpot_ms']:.3f}, "
                f"e2e_ms={r['e2e_ms']:.3f}"
            )

        flush_incremental_buffers()

        if args.benchmark_e2e_txt is not None:
            e2e_vals = load_numeric_lines(args.benchmark_e2e_txt)
            print(f"[OK] Incrementally saved e2e latencies to: {args.benchmark_e2e_txt}")
            print("[Summary][e2e_ms]", json.dumps(summarize_numeric(e2e_vals), indent=2, ensure_ascii=False))

        if args.export_output_lens_txt is not None:
            out_lens = load_numeric_lines(args.export_output_lens_txt)
            print(f"[OK] Incrementally saved output token lengths to: {args.export_output_lens_txt}")
            print("[Summary][output_tokens]", json.dumps(summarize_numeric(out_lens), indent=2, ensure_ascii=False))

        if args.benchmark_json is not None:
            if resume_count > 0:
                print("[WARN] benchmark_json contains only newly processed segment in this run.")
            payload = {
                "args": vars(args),
                "selected_range": {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "final_count": len(prompts),
                    "resume_count": resume_count,
                },
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
