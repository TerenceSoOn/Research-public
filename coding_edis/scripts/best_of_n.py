#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDIS Best-of-N experiment — reproduces Section 5.1 (Figure 4) from the EDIS paper.

Oversampling design (Section 5.1):
  For each question, generate N = m × k total candidates.
  Filter top-k by lowest EDIS score.
  Evaluate three metrics on the EDIS-filtered pool:
    - Avg_EDIS@k:  fraction of filtered k that are correct (pool quality)
    - Best_EDIS@k: is the single lowest-EDIS candidate correct? (best pick)
    - Maj_EDIS@k:  majority vote among filtered k → is that answer correct?

  m ∈ {1, 2, 4, 8, 16}  (oversampling multiplier)
  k = 8                   (candidates to retain)
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from edis.edis_core import compute_edis_from_sequence
from edis.eval_math import extract_answer, is_correct


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_parquet(path: str) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pandas().to_dict(orient="records")


def build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join([m["content"] for m in messages])


# ---------------------------------------------------------------------------
# Entropy from RAW model logits  (Paper Eq. 1)
# ---------------------------------------------------------------------------

def entropy_from_logits(logits: torch.Tensor) -> np.ndarray:
    """
    Compute per-token entropy from the model's raw logit distribution.
    H_t = -Σ_v π_θ(v|x,y_{<t}) log π_θ(v|x,y_{<t})   (Eq. 1)

    Args:
        logits: (T, V) tensor on GPU — raw logits for T tokens.

    Batched on GPU to avoid per-token Python loops and high CPU usage.
    """
    if logits.numel() == 0:
        return np.array([], dtype=np.float32)
    log_probs = torch.log_softmax(logits, dim=-1)   # (T, V) on GPU
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1)           # (T,)
    return ent.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Generation  (batched: generates `batch_size` responses per call)
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> Tuple[List[str], List[np.ndarray], List[float], List[float]]:
    """
    Generate `batch_size` responses and return entropy-based metrics.

    Returns
    -------
    texts          : decoded response strings
    entropies      : per-token entropy trajectories  (from raw logits)
    mean_entropies : mean entropy per response        H̄_i
    sc_scores      : self-certainty per response      log|V| − H̄_i
    """
    input_device = getattr(model, 'device', device)
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    # --- Generate with raw logits (Eq. 1) ---
    try:
        out = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if not hasattr(out, "logits") or out.logits is None:
            raise AttributeError("logits not populated")
        raw_logits = out.logits
    except (TypeError, AttributeError):
        out = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw_logits = tuple(s * temperature for s in out.scores)

    sequences = out.sequences
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    prompt_len = inputs["input_ids"].shape[1]

    vocab_size = model.config.vocab_size
    log_vocab = float(np.log(vocab_size))

    entropies: List[np.ndarray] = []
    mean_entropies: List[float] = []
    sc_scores: List[float] = []

    # Pre-stack logits on GPU: (max_gen_len, batch_size, vocab_size)
    # This avoids per-token GPU tensor access in the loop below.
    stacked_logits = torch.stack(raw_logits)  # stays on GPU

    for i in range(batch_size):
        gen_tokens = sequences[i][prompt_len:].tolist()

        cutoff_index = len(gen_tokens)
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_tokens:
            cutoff_index = gen_tokens.index(tokenizer.eos_token_id) + 1

        valid_logits = stacked_logits[:cutoff_index, i, :]  # (cutoff, V) GPU tensor
        ent_seq = entropy_from_logits(valid_logits)

        entropies.append(ent_seq)
        mean_ent = float(np.mean(ent_seq)) if len(ent_seq) > 0 else 0.0
        mean_entropies.append(mean_ent)
        sc_scores.append(log_vocab - mean_ent)

    return texts, entropies, mean_entropies, sc_scores


def generate_n_total(
    model, tokenizer, prompt: str,
    n_total: int, max_new_tokens: int,
    temperature: float, top_p: float, device: str,
    gen_batch_size: int = 8,
) -> Tuple[List[str], List[np.ndarray], List[float], List[float]]:
    """
    Generate n_total responses, possibly in multiple batches to avoid OOM.
    Returns same format as generate_batch but with n_total items.
    """
    all_texts, all_entropies, all_mean_ent, all_sc = [], [], [], []
    remaining = n_total

    while remaining > 0:
        bs = min(remaining, gen_batch_size)
        texts, entropies, mean_ents, scs = generate_batch(
            model, tokenizer, prompt, bs,
            max_new_tokens, temperature, top_p, device,
        )
        all_texts.extend(texts)
        all_entropies.extend(entropies)
        all_mean_ent.extend(mean_ents)
        all_sc.extend(scs)
        remaining -= bs

    return all_texts, all_entropies, all_mean_ent, all_sc


# ---------------------------------------------------------------------------
# Selection methods  (Section 5.2)
# ---------------------------------------------------------------------------

def top_k_indices(scores: List[float], k: int, lower_is_better: bool) -> List[int]:
    """
    Return indices of the top-k candidates.
    For lower-is-better (EDIS, Entropy): select k candidates with lowest scores.
    For higher-is-better (SC): select k candidates with highest scores.
    """
    indexed = list(enumerate(scores))
    if lower_is_better:
        indexed.sort(key=lambda x: x[1])       # ascending → lowest first
    else:
        indexed.sort(key=lambda x: -x[1])      # descending → highest first
    return [idx for idx, _ in indexed[:k]]


def borda_select(
    texts: List[str],
    scores: List[float],
    lower_is_better: bool = True,
    epsilon: float = 0.1,
) -> int:
    """
    Score-Weighted Borda aggregation  (Section 5.2).

    Each candidate votes for its extracted answer with a confidence weight.
    The answer with the highest total weight wins.

    Weighting (from the paper):
      - Lower-is-better (Entropy, EDIS):  w_i = (s_i + ε)^{-1},  ε = 0.1
      - Higher-is-better (SC):            w_i = s_i

    Returns the index in the input list whose answer matches the winning answer.
    """
    answers = [extract_answer(t) for t in texts]

    if lower_is_better:
        weights = [1.0 / (s + epsilon) for s in scores]
    else:
        weights = [max(s, epsilon) for s in scores]

    # Aggregate weighted votes per unique answer
    answer_votes: Dict[str, float] = {}
    for ans, w in zip(answers, weights):
        answer_votes[ans] = answer_votes.get(ans, 0.0) + w

    best_answer = max(answer_votes, key=answer_votes.get)

    # Among candidates with the winning answer, return the best-weighted one
    best_idx = -1
    best_w = -float("inf")
    for i, (ans, w) in enumerate(zip(answers, weights)):
        if ans == best_answer and w > best_w:
            best_idx = i
            best_w = w

    return best_idx


def select_by_majority_vote(texts: List[str]) -> int:
    """Majority Voting: most-frequent extracted answer."""
    answers = [extract_answer(t) for t in texts]
    counts: Dict[str, List[int]] = {}
    for i, a in enumerate(answers):
        counts.setdefault(a, []).append(i)
    best = max(counts.items(), key=lambda kv: (len(kv[1]), -kv[1][0]))[1]
    return best[0]


def compute_mean_pass_rate(texts: List[str], gts: List[str]) -> float:
    """
    Mean baseline (Table 1 "Mean" column):
    fraction of ALL N candidates that are correct.
    """
    return sum(1 for t in texts if is_correct(t, gts)) / max(len(texts), 1)


# ---------------------------------------------------------------------------
# Oversampling + EDIS filtering pipeline  (Section 5.1, Figure 4)
# ---------------------------------------------------------------------------

def run_selection(
    all_texts: List[str],
    all_edis: List[float],
    gts: List[str],
    k: int,
) -> Dict[str, Any]:
    """
    Section 5.1: EDIS-based best-k-of-N selection.

    Given N = m*k candidates:
    1. Filter top-k by lowest EDIS
    2. Evaluate three metrics on the filtered pool (Figure 4):
       - Avg_EDIS@k:  fraction correct among filtered k
       - Best_EDIS@k: is the single lowest-EDIS candidate correct?
       - Maj_EDIS@k:  majority vote among filtered k
    """
    # --- Baseline: average pass rate across ALL N candidates ---
    baseline_pass = compute_mean_pass_rate(all_texts, gts)

    # --- EDIS filtering: top-k by lowest EDIS ---
    edis_top_k = top_k_indices(all_edis, k, lower_is_better=True)
    filtered_texts = [all_texts[i] for i in edis_top_k]
    filtered_edis = [all_edis[i] for i in edis_top_k]

    # Avg_EDIS@k: pool quality — what fraction of filtered k are correct?
    avg_pass = compute_mean_pass_rate(filtered_texts, gts)

    # Best_EDIS@k: pick the single candidate with lowest EDIS
    best_idx = int(np.argmin(filtered_edis))
    best_correct = int(is_correct(filtered_texts[best_idx], gts))

    # Maj_EDIS@k: majority vote among the filtered k
    maj_idx = select_by_majority_vote(filtered_texts)
    maj_correct = int(is_correct(filtered_texts[maj_idx], gts))

    return {
        "baseline_pass": baseline_pass,
        "avg_edis_pass": avg_pass,
        "best_edis_correct": best_correct,
        "maj_edis_correct": maj_correct,
        "pred_best_edis": filtered_texts[best_idx],
        "pred_maj_edis": filtered_texts[maj_idx],
        "edis_top_k_indices": edis_top_k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDIS Best-of-N Experiment with Oversampling (reproduces Table 1)"
    )
    parser.add_argument(
        "--model",
        default="/home/pterence6699/coding_edis/pretrained/Qwen2.5-math-1.5B",
    )
    parser.add_argument(
        "--dataset",
        default="/home/pterence6699/coding_edis/math_data/math500.parquet",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of candidates to retain after filtering (paper: k=8)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=8,
        help="Oversampling multiplier. Total candidates per question = m * k. "
             "Paper uses m ∈ {1, 2, 4, 8, 16}",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (paper uses pure temperature sampling, i.e. 1.0)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSONL file path")
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=8,
        help="Max sequences to generate per forward pass (to avoid OOM). "
             "Actual total = m*k, generated in chunks of this size.",
    )
    parser.add_argument(
        "--use-multi-gpu",
        action="store_true",
        help="Use device_map='auto' for multi-GPU (requires accelerate)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Limit GPU memory usage fraction (default 0.95)",
    )

    args = parser.parse_args()

    n_total = args.m * args.k
    print(f"Oversampling: m={args.m}, k={args.k}, N=m*k={n_total} candidates per question")

    # Generate default output filename with dataset name
    if args.output is None:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        # Remove trailing numbers like _0726, _1010 for cleaner dir names
        import re
        clean_name = re.sub(r'_\d{4}$', '', dataset_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_str = f"T{args.temperature}".replace(".", "")
        base_name = f"m{args.m}_k{args.k}_{temp_str}_{timestamp}"
        args.output = (
            f"/home/pterence6699/coding_edis/run_result/{clean_name}/jsonl/"
            f"{base_name}.jsonl"
        )
        args._summary_path = (
            f"/home/pterence6699/coding_edis/run_result/{clean_name}/summary/"
            f"{base_name}_summary.txt"
        )

    # ----- Model / tokenizer ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True
    )
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    if args.use_multi_gpu:
        print("Loading model with multi-GPU support (device_map='auto')...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        print(f"Loading model to single GPU: {args.device}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True
        ).to(args.device)

    # Limit GPU memory fraction if requested
    if torch.cuda.is_available() and args.gpu_memory_utilization < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_utilization)
            print(f"Limited GPU memory to {args.gpu_memory_utilization * 100:.1f}%")
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")

    model.eval()

    # ----- Data --------------------------------------------------------------
    rows = load_parquet(args.dataset)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    # ----- Accumulators matching Figure 4 columns ----------------------------
    correct = {"best_edis": 0, "maj_edis": 0}
    baseline_sum = 0.0
    avg_edis_sum = 0.0
    total = 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f_out:
        for row in tqdm(rows, desc=f"Best-of-N (m={args.m}, k={args.k})"):
            messages = row["prompt"]
            prompt = build_prompt(tokenizer, messages)
            gts = list(row["reward_model"]["ground_truth"])

            # --- Generate N = m*k candidates (in batches) ---
            all_texts, all_entropies, all_mean_ent, all_sc = generate_n_total(
                model, tokenizer, prompt,
                n_total=n_total,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                gen_batch_size=args.gen_batch_size,
            )

            # --- Compute EDIS for all N candidates ---
            all_edis = [
                compute_edis_from_sequence(e, return_details=False)
                for e in all_entropies
            ]

            # --- Section 5.1: EDIS filtering + evaluation ---
            result = run_selection(
                all_texts, all_edis, gts, k=args.k
            )

            # --- Accumulate ---
            correct["best_edis"] += result["best_edis_correct"]
            correct["maj_edis"] += result["maj_edis_correct"]
            baseline_sum += result["baseline_pass"]
            avg_edis_sum += result["avg_edis_pass"]
            total += 1

            # --- Per-question record ---
            record = {
                "idx": int(row["extra_info"]["index"]),
                "gts": gts,
                "n_total": n_total,
                "m": args.m,
                "k": args.k,
                "baseline_pass": result["baseline_pass"],
                "avg_edis_pass": result["avg_edis_pass"],
                "best_edis_correct": result["best_edis_correct"],
                "maj_edis_correct": result["maj_edis_correct"],
                "pred_best_edis": result["pred_best_edis"],
                "pred_maj_edis": result["pred_maj_edis"],
                "all_edis_scores": [float(s) for s in all_edis],
                "edis_top_k_indices": result["edis_top_k_indices"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----- Summary ----------------------------------------------------------
    def acc(x):
        return 100.0 * x / max(total, 1)

    baseline_pct = 100.0 * baseline_sum / max(total, 1)
    avg_edis_pct = 100.0 * avg_edis_sum / max(total, 1)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    results_summary = (
        f"Results  ({dataset_name}, m={args.m}, k={args.k}, N={n_total}, "
        f"T={args.temperature}, top_p={args.top_p}):\n"
        f"{'Metric':<20} {'Accuracy':>10}\n"
        f"{'-' * 32}\n"
        f"{'Avg (all N)':<20} {baseline_pct:>9.1f}%\n"
        f"{'Avg_EDIS@' + str(args.k):<20} {avg_edis_pct:>9.1f}%\n"
        f"{'Best_EDIS@' + str(args.k):<20} {acc(correct['best_edis']):>9.1f}%\n"
        f"{'Maj_EDIS@' + str(args.k):<20} {acc(correct['maj_edis']):>9.1f}%\n"
    )

    print(results_summary)

    summary_path = getattr(args, '_summary_path', args.output.replace(".jsonl", "_summary.txt"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        f_sum.write(results_summary)
    print(f"Results saved to {args.output}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
