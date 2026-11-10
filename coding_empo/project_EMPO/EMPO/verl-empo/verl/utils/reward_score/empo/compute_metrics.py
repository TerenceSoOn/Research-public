from typing import List, Optional
import math
import numpy as np
import random

from verl.utils.reward_score.empo.auto_extract import auto_extract
from verl.utils.reward_score.empo.auto_verify import auto_verify

import traceback

def binarize_max(frequencies):
    arr = np.array(frequencies)
    max_val = np.max(arr)  # 获取最大值[7,9](@ref)
    return np.where(arr == max_val, 1, 0).tolist()  # 向量化替换

def semantic_cluster(model_answers, extra_info):
    representatives = []  # 存储独特的答案代表（包括每个空字符串）
    counts = []          # 存储每个独特答案对应的出现次数
    cluster_indices = []  # 存储每个答案对应的聚类索引

    n = len(model_answers)  # 总答案数量
    
    for i, ans in enumerate(model_answers):
        # 处理空字符串：每个空字符串都视为独特聚类
        if ans == "":
            representatives.append(ans)  # 添加空字符串作为新代表
            counts.append(1)              # 出现次数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新聚类索引
            continue
        
        # 处理非空字符串：尝试匹配已有聚类
        found = False
        for idx, rep in enumerate(representatives):
            # 跳过空字符串代表（避免非空字符串与空字符串比较）
            if rep == "":
                continue
            # 使用auto_verify判断答案是否匹配
            rs, _ = auto_verify([ans], [rep], extra_info=extra_info)
            if rs[0]:
                counts[idx] += 1          # 增加计数
                cluster_indices.append(idx)  # 记录聚类索引
                found = True
                #print(f"{[ans]} equal to {[rep]}")
                break
            else:
                #print(f"{[ans]} not equal to {[rep]}")
                pass
        
        # 未找到匹配则创建新聚类
        if not found:
            representatives.append(ans)  # 添加新代表
            counts.append(1)              # 新聚类计数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新索引

    # 计算每个答案的频率（长度为n的列表）
    frequencies = [counts[idx] / n for idx in cluster_indices]
    
    # 计算每个独特答案的频率（长度为len(representatives)的列表）
    unique_frequencies = [c / n for c in counts]
    assert len(representatives) == len(unique_frequencies)
    # print(f"model answers: {model_answers}")
    # print(f"frequencies: {frequencies}")
    # 返回：每个答案的频率列表，所有独特答案列表，独特答案频率列表
    return frequencies, representatives, unique_frequencies

def entropy_thresholding(frequencies, unique_frequencies):
    n = len(unique_frequencies)

    entropy = 0.0
    for p in unique_frequencies:
        if p > 0:
            entropy -= p * math.log(p)
    
    max_entropy = math.log(n)
    
    # min_valid = low * max_entropy
    # max_valid = high * max_entropy
    confidence = max(frequencies)
    
    if confidence >= 0.25:
        return frequencies, 0
    else:
        return [0.0] * len(frequencies), 1

def post_semantic_cluster(model_answers, equality):
    # 存储每个答案的聚类索引（初始为-1）
    cluster_indices = [-1] * len(model_answers)
    # 存储每个聚类的频率（长度等于聚类数量）
    cluster_counts = []
    # 存储每个聚类的代表（第一个加入该聚类的答案）
    representatives = []
    
    n = len(model_answers)
    
    # 遍历每个答案
    for i, ans in enumerate(model_answers):
        if ans == "":
            # 对于空字符串，直接创建新聚类
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)
            continue
            
        # 尝试加入已有聚类（只检查非空聚类）
        found = False
        for j in range(len(cluster_counts)):
            if representatives[j] == "":
                continue  # 跳过空聚类
            rep_index = cluster_indices.index(j)  # 找到该聚类的第一个成员索引
            if equality[i][rep_index]:
                cluster_indices[i] = j
                cluster_counts[j] += 1
                found = True
                break
                
        # 未找到匹配则创建新聚类
        if not found:
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)
    
    # 计算每个答案的频率（长度等于答案数量）
    frequencies = [cluster_counts[idx] / n for idx in cluster_indices]
    # 计算每个聚类的频率（长度等于聚类数量）
    unique_frequencies = [count / n for count in cluster_counts]
    
    return frequencies, representatives, unique_frequencies


def compute_cluster_details(model_answers, equality):
    """Return per-response cluster ids and aggregate cluster stats."""
    cluster_indices = [-1] * len(model_answers)
    cluster_counts = []
    representatives = []

    for i, ans in enumerate(model_answers):
        if ans == "":
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)
            continue

        found = False
        for j in range(len(cluster_counts)):
            if representatives[j] == "":
                continue
            rep_index = cluster_indices.index(j)
            if equality[i][rep_index]:
                cluster_indices[i] = j
                cluster_counts[j] += 1
                found = True
                break

        if not found:
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)

    return {
        "cluster_indices": cluster_indices,
        "cluster_counts": cluster_counts,
        "representatives": representatives,
    }


def compute_vote_artifact(
    cluster_indices: List[int],
    cluster_counts: List[int],
    representatives: List[str],
    vote_mode: str = "majority",
    vote_scores: Optional[List[float]] = None,
    vote_weight_epsilon: float = 0.1,
):
    """
    Build per-response vote metadata from cluster assignments.

    `vote_mode` controls how the "largest/winner" cluster is determined:
      - majority: raw cluster frequency
      - edis_borda: inverse-EDIS weighted cluster vote total

    Winner is always aligned with the active-mode largest cluster.
    """
    n = len(cluster_indices)
    if n == 0 or not cluster_counts:
        return {
            "cluster_id": [],
            "cluster_size": [],
            "cluster_frequency": [],
            "cluster_vote_score": [],
            "sample_vote_weight": [],
            "supports_winner": [],
            "vote_tag": [],
            "winner_cluster_id": -1,
            "winner_label": "",
            "winner_score": 0.0,
            "winner_cluster_frequency": 0.0,
            "largest_cluster_id": -1,
            "in_largest_cluster": [],
        }

    cluster_sizes = [cluster_counts[idx] for idx in cluster_indices]
    cluster_frequencies = [size / n for size in cluster_sizes]

    if vote_mode == "majority":
        sample_vote_weights = [1.0] * n
        cluster_vote_totals = [count / n for count in cluster_counts]
    elif vote_mode == "edis_borda":
        if vote_weight_epsilon <= 0:
            raise ValueError(
                f"vote_weight_epsilon must be positive for vote_mode='edis_borda', got {vote_weight_epsilon}"
            )
        if vote_scores is None:
            raise ValueError("vote_scores must be provided for vote_mode='edis_borda'")
        if len(vote_scores) != n:
            raise ValueError(f"vote_scores length mismatch: {len(vote_scores)} vs {n}")

        sample_vote_weights = []
        for score in vote_scores:
            score = float(score)
            if score < 0:
                raise ValueError(f"EDIS scores must be non-negative, got {score}")
            sample_vote_weights.append(1.0 / (score + vote_weight_epsilon))

        total_weight = sum(sample_vote_weights) or 1.0
        cluster_vote_totals = [0.0] * len(cluster_counts)
        for response_i, cluster_id in enumerate(cluster_indices):
            cluster_vote_totals[cluster_id] += sample_vote_weights[response_i]
        cluster_vote_totals = [score / total_weight for score in cluster_vote_totals]
    else:
        raise ValueError(f"Unsupported vote_mode: {vote_mode}")

    # Weighted-largest semantics:
    # - majority mode: largest by raw cluster frequency
    # - edis_borda mode: largest by weighted cluster vote total
    largest_cluster_id = max(
        range(len(cluster_vote_totals)),
        key=lambda cid: (cluster_vote_totals[cid], cluster_counts[cid], -cid),
    )
    in_largest_cluster = [cluster_id == largest_cluster_id for cluster_id in cluster_indices]

    # Winner is always defined as the active-mode largest cluster.
    winner_cluster_id = largest_cluster_id
    supports_winner = in_largest_cluster
    vote_tags = ["correct" if supports else "incorrect" for supports in supports_winner]
    cluster_vote_scores = [cluster_vote_totals[idx] for idx in cluster_indices]

    return {
        "cluster_id": cluster_indices,
        "cluster_size": cluster_sizes,
        "cluster_frequency": cluster_frequencies,
        "cluster_vote_score": cluster_vote_scores,
        "sample_vote_weight": sample_vote_weights,
        "supports_winner": supports_winner,
        "vote_tag": vote_tags,
        "winner_cluster_id": winner_cluster_id,
        "winner_label": representatives[winner_cluster_id],
        "winner_score": float(cluster_vote_totals[winner_cluster_id]),
        "winner_cluster_frequency": float(cluster_counts[winner_cluster_id] / n),
        "largest_cluster_id": largest_cluster_id,
        "in_largest_cluster": in_largest_cluster,
    }

def run_sequential_auto_verify(
    auto_verify_func,
    model_answers: list[str],
    extra_info=None,
    # timeout and num_workers are not used but kept for signature compatibility
    timeout: int = 60,
    num_workers: int = 1
) -> list[list[int]]:
    """
    Performs sequential (single-process) auto-verification. Useful for debugging.
    """
    N = len(model_answers)
    result_matrix = [[0] * N for _ in range(N)]

    # Use tqdm for a progress bar, as this can be slow
    # pbar = tqdm(total=(N * (N - 1) // 2), desc="Sequential Auto-Verification")
    
    for i in range(N):
        result_matrix[i][i] = 1 # An answer is always equivalent to itself
        for j in range(i + 1, N):
            try:
                # Directly call the function
                result = auto_verify_func([model_answers[i]], [model_answers[j]], extra_info)
                
                # Handle both tuple and int return types
                value = result[0][0] if isinstance(result, tuple) else result
                result_matrix[i][j] = value
                result_matrix[j][i] = value # Similarity Matrix is symmetric

            except Exception:
                e = traceback.format_exc()
                print(f"\n[Error] Verification for pair ({i}, {j}) failed: {e}")
                result_matrix[i][j] = 0
                result_matrix[j][i] = 0
                
    return result_matrix


def empo_metrics(
    solutions: List[str],
    ground_truth: List[str],
    extra_info=None,
    reward_type=None,
    entropy_thres=None,
    vote_scores: Optional[List[float]] = None,
    vote_mode: str = "majority",
    vote_weight_epsilon: float = 0.1,
):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    if isinstance(ground_truth[0], list):
        ground_truth = [gt[-1] for gt in ground_truth]
    assert (
        isinstance(ground_truth, list) and
        all(isinstance(item, str) for item in ground_truth)
    ), "Ground truth must be list[str]"
    
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {set(ground_truth)}"
    ground_truth_str = ground_truth[0]
    
    model_answers = auto_extract(solutions, extra_info=extra_info)
    assert (
        isinstance(model_answers, list) and
        all(isinstance(item, str) for item in model_answers)
    ), "Model answers must be list[str]"

    answers_with_gt = model_answers + [ground_truth_str]
    result_matrix = run_sequential_auto_verify(auto_verify, answers_with_gt, extra_info=extra_info)
    equality_matrix = [row[:-1] for row in result_matrix[:-1]]
    true_rewards = result_matrix[-1][:-1]
    assert len(true_rewards) == len(model_answers)

    cluster_details = compute_cluster_details(model_answers, equality_matrix)
    cluster_indices = cluster_details["cluster_indices"]
    cluster_counts = cluster_details["cluster_counts"]
    unique_answers = cluster_details["representatives"]
    n_answers = len(model_answers) or 1
    frequencies = [cluster_counts[idx] / n_answers for idx in cluster_indices]
    unique_frequencies = [count / n_answers for count in cluster_counts]
    vote_artifact = compute_vote_artifact(
        cluster_indices=cluster_indices,
        cluster_counts=cluster_counts,
        representatives=unique_answers,
        vote_mode=vote_mode,
        vote_scores=vote_scores,
        vote_weight_epsilon=vote_weight_epsilon,
    )
    
    # Handle case where frequencies is empty
    if not frequencies:
        hit_rate = 0.0
        majority_ratio = 0.0
        estimated_label = ""
    else:
        winner_cluster_id = vote_artifact["winner_cluster_id"]
        winner_indices = [i for i, cluster_id in enumerate(cluster_indices) if cluster_id == winner_cluster_id]
        max_index = winner_indices[0]
        estimated_label = vote_artifact["winner_label"]
        majority_ratio = vote_artifact["winner_cluster_frequency"]
        hit_rate = float(true_rewards[max_index])

    # Support hybrid reward types combining entropy (semantic agreement) and ground-truth correctness.
    # Format: "hybrid[:alpha:beta]" where alpha weights entropy frequency and beta weights correctness (gt).
    # Example: hybrid:0.7:0.3 => reward = 0.7 * freq + 0.3 * gt
    is_hybrid = isinstance(reward_type, str) and reward_type.startswith('hybrid')
    if not is_hybrid:
        assert reward_type in ['gt', 'entropy', 'voting', 'format', 'random', 'best'], f"Unsupported reward_type {reward_type}"
    if reward_type == 'voting':
        rewards = [1.0 if supports else 0.0 for supports in vote_artifact["supports_winner"]]
    elif reward_type == 'entropy':
        # rewards, filtered = entropy_thresholding(frequencies, unique_frequencies)
        rewards = frequencies #, unique_frequencies
    elif reward_type == 'gt':
        rewards = true_rewards
    elif reward_type == 'best':
        rewards = true_rewards if hit_rate < 1 else [0.0] * len(true_rewards)
    elif reward_type == 'random':
        rewards = [random.uniform(0, 1) for _ in range(len(true_rewards))]
    elif reward_type == 'format':
        rewards = [1.0 if len(ans) > 0 else 0 for ans in model_answers]
    elif is_hybrid:
        # Parse weights
        alpha, beta = 0.5, 0.5  # defaults
        parts = reward_type.split(':')
        if len(parts) == 3:
            try:
                alpha = float(parts[1]); beta = float(parts[2])
            except ValueError:
                print(f"[Hybrid Reward] Failed to parse alpha/beta from {reward_type}, using defaults 0.5/0.5")
        elif len(parts) != 1:
            print(f"[Hybrid Reward] Unexpected hybrid format {reward_type}, expected hybrid[:alpha:beta]. Using defaults 0.5/0.5")

        # Combine entropy frequency (already in [0,1]) and correctness (0/1)
        rewards = [alpha * f + beta * tr for f, tr in zip(frequencies, true_rewards)]
        # Optional clip to [0,1]
        rewards = [min(1.0, max(0.0, r)) for r in rewards]
        
        # Debug logging for hybrid reward
        if random.random() < 0.01: # Log 1% of the time to avoid spam
             print(f"[Hybrid Debug] Type: {reward_type}, Alpha: {alpha}, Beta: {beta}")
             print(f"[Hybrid Debug] Sample Freq: {frequencies[0]:.4f}, GT: {true_rewards[0]}, Reward: {rewards[0]:.4f}")

        # For metric tracking
        hybrid_weights = (alpha, beta)
    else:
        # Should be unreachable due to earlier assert
        rewards = true_rewards

    rewards_hit_rate = sum(1 for r, tr in zip(rewards, true_rewards) if r == tr) / len(rewards) if rewards else 0

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    metrics = {
        "label_accuracy": [hit_rate] * len(model_answers),
        "reward_accuracy": [rewards_hit_rate] * len(model_answers),
        "majority_ratio": [majority_ratio] * len(model_answers),
        "train_accuracy": [sum(true_rewards) / len(true_rewards) if true_rewards else 0.0] * len(model_answers),
        "train_reward": [sum(rewards) / len(rewards) if rewards else 0.0] * len(model_answers),
        f"pass@{len(solutions)}": [1.0 if sum(true_rewards) >= 1 else 0.0] * len(model_answers),
        "extracted_answers": model_answers,
        "estimated_label": [estimated_label] * len(model_answers),
        "cluster_id": vote_artifact["cluster_id"],
        "cluster_size": vote_artifact["cluster_size"],
        "cluster_frequency": vote_artifact["cluster_frequency"],
        "cluster_vote_score": vote_artifact["cluster_vote_score"],
        "sample_vote_weight": vote_artifact["sample_vote_weight"],
        "vote_supports_winner": vote_artifact["supports_winner"],
        "vote_tag": vote_artifact["vote_tag"],
        "vote_winner_label": [vote_artifact["winner_label"]] * len(model_answers),
        "vote_winner_score": [vote_artifact["winner_score"]] * len(model_answers),
        "vote_mode": [vote_mode] * len(model_answers),
        "largest_cluster_id": [vote_artifact["largest_cluster_id"]] * len(model_answers),
        "in_largest_cluster": vote_artifact["in_largest_cluster"],
    }

    if is_hybrid:
        metrics["hybrid_alpha"] = [hybrid_weights[0]] * len(model_answers)
        metrics["hybrid_beta"] = [hybrid_weights[1]] * len(model_answers)
        metrics["component_entropy_reward_mean"] = [float(sum(frequencies)/len(frequencies)) if frequencies else 0.0] * len(model_answers)
        metrics["component_gt_reward_mean"] = [float(sum(true_rewards)/len(true_rewards)) if true_rewards else 0.0] * len(model_answers)

    return rewards, metrics
