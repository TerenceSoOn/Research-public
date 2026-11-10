#!/usr/bin/env python3
"""
Step 1 of the split precompute pipeline: generate responses and extract answers.

Generates N responses per prompt from the frozen reference model using vLLM,
runs auto_extract on each response, and saves the extracted answers to an
intermediate JSONL file for downstream clustering.

Output format (one JSON line per prompt):
    {"index": 0, "prompt": ..., "extra_info": ..., "reward_model": ...,
     "extracted_answers": ["ans1", "ans2", ...]}
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser(description="Generate responses and extract answers")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_path", type=str, default=None,
                    help="Output JSONL (default: <data_stem>.extracted_answers.<model_tag>.jsonl)")
    p.add_argument("--n_responses", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_response_length", type=int, default=3072)
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--model_impl", type=str, default="vllm", choices=["vllm", "transformers"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_prompts", type=int, default=None,
                    help="Process only the first N prompts (default: all)")
    return p.parse_args()


def load_data(data_path: str) -> list[dict]:
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt_texts(records, tokenizer):
    prompts = []
    for r in records:
        prompt = r["prompt"]
        if isinstance(prompt, list):
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback for tokenizers without built-in chat template.
                chunks = []
                for msg in prompt:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    chunks.append(f"{role}: {content}")
                text = "\n".join(chunks) + "\nassistant:"
        else:
            text = prompt
        prompts.append(text)
    return prompts


def generate_with_transformers(args, prompts, tokenizer):
    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32
    model = None
    # Mistral3-family may not map to AutoModelForCausalLM in some versions.
    for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            kwargs = dict(
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if has_cuda and torch.cuda.device_count() > 1:
                # Shard model across visible GPUs for faster generation.
                kwargs["device_map"] = "auto"
            model = cls.from_pretrained(args.model_path, **kwargs)
            if not (has_cuda and torch.cuda.device_count() > 1):
                device = "cuda" if has_cuda else "cpu"
                model = model.to(device)
            break
        except Exception:
            model = None
    if model is None:
        raise RuntimeError("Failed to load model with both AutoModelForImageTextToText and AutoModelForCausalLM.")
    model.eval()
    outputs = []
    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_response_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start:batch_start + args.batch_size]
        batch = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        if has_cuda and torch.cuda.device_count() > 1:
            # Place inputs on the first shard device for dispatch.
            first_device = next(model.parameters()).device
            batch = {k: v.to(first_device) for k, v in batch.items()}
        else:
            device = "cuda" if has_cuda else "cpu"
            batch = {k: v.to(device) for k, v in batch.items()}
        bsz = batch["input_ids"].shape[0]
        input_ids = batch["input_ids"].repeat_interleave(args.n_responses, dim=0)
        attention_mask = batch["attention_mask"].repeat_interleave(args.n_responses, dim=0)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        prompt_lens = attention_mask.sum(dim=1).tolist()
        texts = []
        for seq, plen in zip(generated, prompt_lens):
            texts.append(tokenizer.decode(seq[int(plen):], skip_special_tokens=True))
        for i in range(bsz):
            group = texts[i * args.n_responses:(i + 1) * args.n_responses]
            outputs.append(group)
    return outputs


def main():
    args = parse_args()

    if args.output_path is None:
        stem = Path(args.data_path).stem
        args.output_path = str(
            Path(args.data_path).parent / f"{stem}.extracted_answers.jsonl"
        )

    tp_size = args.tensor_parallel_size or torch.cuda.device_count()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = None
    if args.model_impl == "vllm":
        print(f"Initializing vLLM with {tp_size} GPUs, model: {args.model_path}")
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=tp_size,
            seed=args.seed,
            max_model_len=args.max_prompt_length + args.max_response_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
    else:
        print(f"Using transformers backend, model: {args.model_path}")
    records = load_data(args.data_path)
    if args.max_prompts is not None:
        records = records[:args.max_prompts]
    prompts = build_prompt_texts(records, tokenizer)

    sampling_params = SamplingParams(
        n=args.n_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )

    total_prompts = len(prompts)
    print(f"Generating {args.n_responses} responses for {total_prompts} prompts "
          f"(temperature={args.temperature}, top_p={args.top_p})")
    print(f"Output: {args.output_path}")

    all_outputs = []
    t_start = time.time()

    if args.model_impl == "vllm":
        for batch_start in range(0, total_prompts, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_prompts)
            batch_prompts = prompts[batch_start:batch_end]
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(batch_outputs)

            elapsed = time.time() - t_start
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_prompts - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total_prompts}] {elapsed:.0f}s elapsed, "
                  f"{rate:.1f} prompts/s, ETA {eta:.0f}s")
    else:
        grouped = generate_with_transformers(args, prompts, tokenizer)
        class _Output:
            def __init__(self, texts):
                self.outputs = [type("O", (), {"text": t})() for t in texts]
        all_outputs = [_Output(g) for g in grouped]

    t_gen = time.time() - t_start
    print(f"Generation complete in {t_gen:.0f}s "
          f"({total_prompts * args.n_responses} total responses)")

    from verl.utils.reward_score.empo.auto_extract import auto_extract

    print("Extracting answers...")
    n_written = 0
    with open(args.output_path, "w") as fout:
        for i, (record, output) in enumerate(tqdm(
            zip(records, all_outputs), total=len(records), desc="Extracting"
        )):
            responses = [o.text for o in output.outputs]
            extra_info = record.get("extra_info", {})
            extracted = auto_extract(responses, extra_info=extra_info)

            out_record = {
                "index": i,
                "prompt": record["prompt"],
                "extra_info": record.get("extra_info", {}),
                "reward_model": record.get("reward_model", {}),
                "extracted_answers": extracted,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            n_written += 1

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Prompts:    {total_prompts}")
    print(f"  Responses:  {total_prompts * args.n_responses}")
    print(f"  Gen time:   {t_gen:.0f}s")
    print(f"  Total time: {t_total:.0f}s")
    print(f"  Written:    {n_written}")
    print(f"  Output:     {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
