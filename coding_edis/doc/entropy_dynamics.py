# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Entropy Dynamics utilities for validation-time filtering.

Based on the observation that wrong responses have significantly higher entropy dynamics:
- 1.69x higher entropy variance
- 1.58x higher mean entropy  
- 1.17x higher spike rate

We propose the Entropy Dynamics Instability Score (EDIS) for filtering.
"""

from typing import Optional
import numpy as np
import torch


def compute_spike_count(
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 0.3,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> int:
    """
    Count the number of entropy spikes (sudden increases > threshold).
    This is the SIMPLE/LEGACY method, kept for backward compatibility.
    
    Args:
        entropy_sequence: 1D array of entropy values for each token position.
        threshold: Minimum entropy increase to count as a spike. Default 0.3.
        mask: Optional mask indicating valid tokens (1=valid, 0=padding).
    
    Returns:
        Number of entropy spikes in the sequence.
    """
    if isinstance(entropy_sequence, torch.Tensor):
        entropy_sequence = entropy_sequence.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if len(entropy_sequence) < 2:
        return 0
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]
    
    # Compute differences between consecutive entropy values
    diffs = np.diff(entropy_sequence)
    
    # Count spikes (entropy increases > threshold)
    spike_count = np.sum(diffs > threshold)
    
    return int(spike_count)


def compute_burst_spike(
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 1.3,
    window: int = 5,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> int:
    """
    Count burst spikes: cumulative entropy increase within a sliding window.
    
    Detects "persistent deterioration" - entropy continuously rising over multiple tokens,
    indicating the model is getting progressively more confused.
    
    Args:
        entropy_sequence: 1D array of entropy values for each token position.
        threshold: Cumulative increase threshold. Default 1.3.
        window: Sliding window size. Default 5 tokens.
        mask: Optional mask indicating valid tokens (1=valid, 0=padding).
    
    Example (threshold=1.3, window=5):
        Sequence: [1.0, 1.2, 1.4, 1.6, 1.8, 2.4]
        Position 0: seq[5] - seq[0] = 2.4 - 1.0 = 1.4 > 1.3 → spike!
    
    Returns:
        Number of burst spikes in the sequence.
    """
    if isinstance(entropy_sequence, torch.Tensor):
        entropy_sequence = entropy_sequence.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < window + 1:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < window + 1:
        return 0
    
    # Count positions where cumulative increase exceeds threshold
    count = 0
    for i in range(len(entropy_sequence) - window):
        cumulative_increase = entropy_sequence[i + window] - entropy_sequence[i]
        if cumulative_increase > threshold:
            count += 1
    
    return count


def compute_peak_spike(
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 1.3,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> int:
    """
    Count peak-to-trough spikes: rebound from historical minimum.
    
    Detects "V-shaped recovery" patterns - when entropy rebounds from its 
    historical low point by more than threshold. This captures moments where
    the model thought it understood but then got confused again.
    
    Args:
        entropy_sequence: 1D array of entropy values for each token position.
        threshold: Rebound threshold from historical minimum. Default 1.3.
        mask: Optional mask indicating valid tokens (1=valid, 0=padding).
    
    Example (threshold=1.3):
        Sequence: [2.0, 1.5, 1.0, 0.5, 0.8, 1.5, 2.0]
        At position 5: min so far = 0.5, current = 1.5, rebound = 1.0 < 1.3 → no spike
        At position 6: min so far = 0.5, current = 2.0, rebound = 1.5 > 1.3 → spike!
    
    Returns:
        Number of peak-to-trough spikes in the sequence.
    """
    if isinstance(entropy_sequence, torch.Tensor):
        entropy_sequence = entropy_sequence.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < 2:
        return 0
    
    # Track historical minimum and count rebounds
    count = 0
    historical_min = entropy_sequence[0]
    
    for i in range(1, len(entropy_sequence)):
        current = entropy_sequence[i]
        # Check if current rebounds from historical minimum by more than threshold
        rebound = current - historical_min
        if rebound > threshold:
            count += 1
        # Update historical minimum
        if current < historical_min:
            historical_min = current
    
    return count


def compute_hybrid_spike(
    entropy_sequence: np.ndarray | torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> tuple[float, int, int]:
    """
    Compute hybrid spike score: 0.5 × (burst + peak_trough).
    
    Combines two complementary metrics:
    - Burst spike: seq[i+window] - seq[i] > burst_threshold (持续恶化)
    - Peak-to-Trough spike: rebound from historical minimum > peak_threshold (V型反弹)
    
    Formula: spike = 0.5 × (burst_spike + peak_trough_spike)
    
    Args:
        entropy_sequence: 1D array of entropy values.
        burst_threshold: Cumulative increase threshold in window (default 1.5).
        peak_threshold: Rebound from historical minimum threshold (default 1.3).
        window: Sliding window size for burst detection (default 5).
        mask: Optional mask for valid tokens.
    
    Returns:
        Tuple of (combined_spike, burst_spike, peak_trough_spike)
    """
    burst_spike = compute_burst_spike(
        entropy_sequence, 
        threshold=burst_threshold,
        window=window,
        mask=mask
    )
    
    peak_spike = compute_peak_spike(
        entropy_sequence, 
        threshold=peak_threshold, 
        mask=mask
    )
    
    # Average of two methods: interpretable as "average entropy spike count"
    combined_spike = 0.5 * (burst_spike + peak_spike)
    
    return combined_spike, burst_spike, peak_spike


def compute_entropy_variance(
    entropy_sequence: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> float:
    """
    Compute the variance of entropy values.
    
    Args:
        entropy_sequence: 1D array of entropy values for each token position.
        mask: Optional mask indicating valid tokens (1=valid, 0=padding).
    
    Returns:
        Variance of entropy values.
    """
    if isinstance(entropy_sequence, torch.Tensor):
        entropy_sequence = entropy_sequence.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if len(entropy_sequence) < 2:
        return 0.0
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]
    
    return float(np.var(entropy_sequence))


def compute_mean_entropy(
    entropy_sequence: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None
) -> float:
    """
    Compute the mean entropy value.
    
    Args:
        entropy_sequence: 1D array of entropy values for each token position.
        mask: Optional mask indicating valid tokens (1=valid, 0=padding).
    
    Returns:
        Mean entropy value.
    """
    if isinstance(entropy_sequence, torch.Tensor):
        entropy_sequence = entropy_sequence.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if len(entropy_sequence) < 1:
        return 0.0
    
    # Apply mask if provided
    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 1:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]
    
    return float(np.mean(entropy_sequence))


def compute_edis(
    spike_count: float,  # Can be float for hybrid method
    entropy_variance: float,
    mean_entropy: float = None,  # Not used in current formula
    seq_length: int = None,       # Not used in current formula
    **kwargs  # For backward compatibility
) -> float:
    """
    Compute Entropy Dynamics Instability Score (EDIS).
    
    EDIS quantifies response quality based on entropy dynamics ("熵突变").
    Higher EDIS indicates more unstable generation, likely incorrect.
    
    Formula:
        EDIS = spike_count × (1 + variance)
    
    Design rationale:
        - spike_count: Combined spike score (burst + peak for hybrid method)
          NOT normalized by length - longer wrong responses have more spikes!
        - variance: Entropy fluctuation magnitude
        - Multiplicative form: spikes and variance reinforce each other
    
    Empirical performance (hybrid burst + peak-to-trough method):
        - Discrimination ratio: 1.84x (wrong/correct EDIS)
        - Filtering accuracy: 52.2% (vs 32.3% baseline)
        - Best params: burst=1.5 (window=5), peak=1.3 (rebound from min)
    
    Args:
        spike_count: Spike score (int for simple, float for hybrid)
        entropy_variance: Variance of entropy sequence
        mean_entropy: (unused) Kept for API compatibility
        seq_length: (unused) Kept for API compatibility
    
    Returns:
        EDIS score (lower is better, indicates more stable generation)
    """
    # EDIS = spike_count × (1 + variance)
    # Simple, interpretable, and empirically optimal
    edis = spike_count * (1 + entropy_variance)
    
    return float(edis)


def compute_entropy_metrics(
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
    **kwargs  # For backward compatibility
) -> dict:
    """
    Compute entropy dynamics metrics for a batch of responses.
    
    Uses the hybrid (burst + peak-to-trough) method:
    - spike = 0.5 × (burst_spike + peak_trough_spike)
    - EDIS = spike × (1 + variance)
    
    Args:
        entropys: Tensor of shape [batch_size, response_length] with entropy values.
        response_masks: Tensor of shape [batch_size, response_length] indicating valid tokens.
        burst_threshold: Cumulative increase threshold for burst detection. Default 1.5.
        peak_threshold: Rebound threshold for peak-to-trough detection. Default 1.3.
        window: Sliding window size for burst detection. Default 5.
    
    Returns:
        Dictionary containing:
            - mean_entropys: List of mean entropy for each response
            - spike_counts: List of spike scores (0.5 × (burst + peak))
            - entropy_variances: List of entropy variances for each response
            - edis_scores: EDIS scores for filtering (lower is better)
            - combined_scores: Normalized EDIS scores [0, 1]
            - burst_spikes: List of burst spike counts
            - peak_spikes: List of peak spike counts
    """
    batch_size = entropys.shape[0]
    
    mean_entropys = []
    spike_counts = []  # 0.5 × (burst + peak)
    entropy_variances = []
    burst_spikes = []
    peak_spikes = []
    
    for i in range(batch_size):
        entropy_seq = entropys[i]
        mask = response_masks[i] if response_masks is not None else None
        
        mean_entropy = compute_mean_entropy(entropy_seq, mask=mask)
        entropy_var = compute_entropy_variance(entropy_seq, mask=mask)
        
        # Use burst + peak hybrid method
        combined_spike, burst_spike, peak_spike = compute_hybrid_spike(
            entropy_seq,
            burst_threshold=burst_threshold,
            peak_threshold=peak_threshold,
            window=window,
            mask=mask
        )
        
        mean_entropys.append(mean_entropy)
        spike_counts.append(combined_spike)
        entropy_variances.append(entropy_var)
        burst_spikes.append(burst_spike)
        peak_spikes.append(peak_spike)
    
    # Compute EDIS for each sample
    # EDIS = spike × (1 + variance)
    edis_scores = []
    for i in range(batch_size):
        edis = compute_edis(
            spike_count=spike_counts[i],
            entropy_variance=entropy_variances[i]
        )
        edis_scores.append(edis)
    
    # Normalize EDIS to [0, 1] for combined_scores
    edis_arr = np.array(edis_scores, dtype=np.float32)
    if edis_arr.max() > edis_arr.min():
        combined_scores = (edis_arr - edis_arr.min()) / (edis_arr.max() - edis_arr.min())
    else:
        combined_scores = np.zeros_like(edis_arr)
    
    return {
        "mean_entropys": mean_entropys,
        "spike_counts": spike_counts,
        "entropy_variances": entropy_variances,
        "edis_scores": edis_scores,
        "combined_scores": combined_scores.tolist(),
        "burst_spikes": burst_spikes,
        "peak_spikes": peak_spikes,
    }


def filter_by_entropy_metrics(
    indices: list[int],
    spike_counts: list[int],
    entropy_variances: list[float],
    n_keep_per_group: int,
    uid_to_indices: Optional[dict] = None,
    mean_entropys: Optional[list[float]] = None,  # Kept for API compatibility
    seq_lengths: Optional[list[int]] = None,       # Kept for API compatibility
    edis_weights: Optional[dict] = None            # Kept for API compatibility
) -> list[int]:
    """
    Filter responses by EDIS (Entropy Dynamics Instability Score).
    
    Keeps top n_keep_per_group samples with LOWEST EDIS (most stable generation).
    
    EDIS Formula: spike_count × (1 + variance)
    
    Args:
        indices: Original indices of responses.
        spike_counts: List of spike counts for each response.
        entropy_variances: List of entropy variances for each response.
        n_keep_per_group: Number of responses to keep per group.
        uid_to_indices: Optional dict mapping uid to list of indices (for per-prompt filtering).
        mean_entropys: (unused) Kept for backward compatibility.
        seq_lengths: (unused) Kept for backward compatibility.
        edis_weights: (unused) Kept for backward compatibility.
    
    Returns:
        List of indices to keep (those with lowest EDIS).
    """
    def compute_group_edis(group_indices):
        """Compute EDIS for a group of samples."""
        edis_scores = []
        for i in group_indices:
            edis = compute_edis(
                spike_count=spike_counts[i],
                entropy_variance=entropy_variances[i]
            )
            edis_scores.append(edis)
        return edis_scores
    
    if uid_to_indices is not None:
        # Filter within each group (same prompt)
        keep_indices = []
        for uid, group_indices in uid_to_indices.items():
            edis_scores = compute_group_edis(group_indices)
            
            # Sort by EDIS (lower is better) and keep top n_keep_per_group
            n_keep = min(n_keep_per_group, len(group_indices))
            sorted_local_indices = np.argsort(edis_scores)[:n_keep]
            keep_indices.extend([group_indices[i] for i in sorted_local_indices])
        
        return keep_indices
    else:
        # Global filtering (no grouping)
        edis_scores = compute_group_edis(indices)
        
        # Sort and keep top n_keep_per_group
        sorted_indices = np.argsort(edis_scores)[:n_keep_per_group]
        return [indices[i] for i in sorted_indices]


def group_indices_by_uid(uids: list | np.ndarray) -> dict:
    """
    Group indices by uid.
    
    Args:
        uids: List or array of uids.
    
    Returns:
        Dictionary mapping uid to list of indices.
    """
    uid_to_indices = {}
    for i, uid in enumerate(uids):
        if uid not in uid_to_indices:
            uid_to_indices[uid] = []
        uid_to_indices[uid].append(i)
    return uid_to_indices


# ============================================================================
# EDIS-based Advantage Weighting for Training (Method C)
# ============================================================================

def compute_edis_weights(
    edis_scores: torch.Tensor | np.ndarray | list,
    is_correct: torch.Tensor | np.ndarray | list,
    alpha: float = 1.0,
    group_indices: Optional[list[list[int]]] = None
) -> tuple[torch.Tensor, dict]:
    """
    Compute advantage weights based on EDIS and correctness using Log-Softmax.
    
    Uses Log-Softmax with correct/wrong group normalization to maintain
    advantage mean ≈ 0 while emphasizing boundary samples.
    
    Core insight (反转版四象限策略):
        - 低EDIS+正确 = "已掌握" → 权重高 (强化巩固)
        - 高EDIS+错误 = "挣扎失败" → 权重高 (惩罚学习)
        - 高EDIS+正确 = 可能是运气 → 权重低
        - 低EDIS+错误 = 系统性错误 → 权重低 (让adv自然惩罚)
    
    Formula (Log-Softmax with group normalization):
        1. log_edis = log(edis + 1)  # Log transform for symmetric distribution
        2. centered = log_edis - mean(log_edis)  # Center at mean
        3. signed = correct ? -centered : centered  # 反转版: 低正+高错→高权重
        4. k = alpha * std(log_edis)  # Temperature based on data spread
        5. weights = softmax(signed / k) * n  # Softmax weighting
        6. Normalize within correct/wrong groups separately  # Maintain adv mean ≈ 0
    
    Why Log transform:
        - Makes EDIS distribution more symmetric (skewness 2.11 → -0.48)
        - Better target/non-target discrimination (2.79x → 3.72x)
        - Reduces extreme weight ratios (1693x → 108x)
    
    Why group normalization:
        - Correct samples: total weight = n_correct
        - Wrong samples: total weight = n_wrong
        - Ensures weighted advantage mean ≈ 0 (balanced gradients)
    
    Args:
        edis_scores: EDIS scores for each sample. Shape: [batch_size]
        is_correct: Boolean or float tensor indicating correctness. Shape: [batch_size]
        alpha: Softmax temperature multiplier (controls weight distribution shape).
               温度系数，控制权重分布的集中/分散程度。
               
               k = alpha * std(log_edis)
               
               Effect on softmax(x/k):
               - 小 alpha → 小 k → x/k 更大 → softmax 更尖锐 → 权重更集中
               - 大 alpha → 大 k → x/k 更小 → softmax 更平缓 → 权重更均匀
               
               实际效果 (目标样本放大倍数):
               - alpha=0.5: 2-5x   (激进，少数样本主导梯度)
               - alpha=1.0: 2-2.5x (推荐，平衡区分度和稳定性)
               - alpha=2.0: 1.5-2x (保守，更均匀的权重分布)
               
        group_indices: List of index lists for each group. If None, treat all as one group.
    
    Returns:
        Tuple of (weights, metrics_dict)
        - weights: EDIS-based weights. Shape: [batch_size], mean=1.0
        - metrics_dict: Dictionary of metrics for logging
    
    Weight distribution (alpha=1.0, typical data):
        - 高EDIS+正确: ~2.0x (within correct group)
        - 低EDIS+正确: ~0.6x (within correct group)
        - 低EDIS+错误: ~2.4x (within wrong group)
        - 高EDIS+错误: ~0.6x (within wrong group)
    """
    # Convert to tensors
    if isinstance(edis_scores, list):
        edis_scores = torch.tensor(edis_scores, dtype=torch.float32)
    elif isinstance(edis_scores, np.ndarray):
        edis_scores = torch.from_numpy(edis_scores).float()
    
    if isinstance(is_correct, list):
        is_correct = torch.tensor(is_correct, dtype=torch.float32)
    elif isinstance(is_correct, np.ndarray):
        is_correct = torch.from_numpy(is_correct).float()
    elif isinstance(is_correct, torch.Tensor):
        is_correct = is_correct.float()
    
    device = edis_scores.device if isinstance(edis_scores, torch.Tensor) else 'cpu'
    edis_scores = edis_scores.to(device)
    is_correct = is_correct.to(device)
    
    batch_size = len(edis_scores)
    weights = torch.ones(batch_size, dtype=torch.float32, device=device)
    
    # Metrics to collect
    metrics = {}
    
    # Track mixed vs non-mixed groups
    n_groups_total = 0
    n_groups_mixed = 0
    n_groups_all_correct = 0
    n_groups_all_wrong = 0
    n_samples_weighted = 0
    n_samples_skipped = 0
    
    # If no group_indices, treat all samples as one group
    if group_indices is None:
        group_indices = [list(range(batch_size))]
    
    for indices in group_indices:
        if len(indices) == 0:
            continue
        
        n_groups_total += 1
        indices_t = torch.tensor(indices, dtype=torch.long, device=device)
        group_edis = edis_scores[indices_t]
        group_correct = is_correct[indices_t]
        n = len(indices)
        
        # Check if this is a mixed group (has both correct and wrong samples)
        n_correct_in_group = (group_correct > 0.5).sum().item()
        n_wrong_in_group = n - n_correct_in_group
        
        # Only apply EDIS weighting to mixed groups
        if n_correct_in_group == 0:
            # All wrong - skip EDIS weighting, keep weights = 1.0
            n_groups_all_wrong += 1
            n_samples_skipped += n
            continue
        elif n_wrong_in_group == 0:
            # All correct - skip EDIS weighting, keep weights = 1.0
            n_groups_all_correct += 1
            n_samples_skipped += n
            continue
        
        # This is a mixed group - apply EDIS weighting
        n_groups_mixed += 1
        n_samples_weighted += n
        
        # Step 1: Log transform (makes distribution more symmetric)
        log_edis = torch.log(group_edis + 1)
        
        # Step 2: Center at mean (log后近似对称，mean更优)
        mean = log_edis.mean()
        centered = log_edis - mean
        
        # Step 3: 反转版 - 根据对错翻转符号
        # 低EDIS+正确 → 高权重（强化已掌握的知识）
        # 高EDIS+错误 → 高权重（惩罚挣扎仍失败的）
        # 高EDIS+正确 → 低权重（可能是运气）
        # 低EDIS+错误 → 低权重（系统性错误，让advantage负值自然惩罚）
        signed = torch.where(group_correct > 0.5, -centered, centered)
        
        # Step 4: Compute temperature k = alpha * std
        log_std = log_edis.std()
        if log_std < 1e-8:
            # Edge case: all EDIS are the same
            weights[indices_t] = 1.0
            continue
        
        k = alpha * log_std
        
        # Step 5: Softmax weighting
        # softmax(x) = exp(x - max) / sum(exp(x - max))
        signed_scaled = signed / k
        exp_vals = torch.exp(signed_scaled - signed_scaled.max())
        softmax_weights = exp_vals / exp_vals.sum() * n
        
        # Step 6: Normalize within correct/wrong groups separately
        # This ensures: sum(weights[correct]) = n_correct, sum(weights[wrong]) = n_wrong
        # So weighted advantage mean ≈ 0 (balanced positive/negative gradients)
        correct_mask = group_correct > 0.5
        wrong_mask = ~correct_mask
        n_correct = correct_mask.sum()
        n_wrong = wrong_mask.sum()
        
        group_weights = torch.zeros_like(softmax_weights)
        
        if n_correct > 0:
            correct_weights = softmax_weights[correct_mask]
            correct_weights = correct_weights / correct_weights.sum() * n_correct
            group_weights[correct_mask] = correct_weights
        
        if n_wrong > 0:
            wrong_weights = softmax_weights[wrong_mask]
            wrong_weights = wrong_weights / wrong_weights.sum() * n_wrong
            group_weights[wrong_mask] = wrong_weights
        
        # Assign weights back
        weights[indices_t] = group_weights
        
        # Collect group-level metrics (use last group if multiple)
        high_edis = log_edis > mean
        hc = high_edis & correct_mask  # 高EDIS+正确
        lc = (~high_edis) & correct_mask  # 低EDIS+正确
        hw = high_edis & wrong_mask  # 高EDIS+错误
        lw = (~high_edis) & wrong_mask  # 低EDIS+错误
        
        # === Step 1: Log transform metrics ===
        metrics["edis_weight/log_edis_mean"] = log_edis.mean().item()
        metrics["edis_weight/log_edis_std"] = log_std.item()
        metrics["edis_weight/log_edis_center"] = mean.item()  # 中心点（现在用mean）
        metrics["edis_weight/log_edis_min"] = log_edis.min().item()
        metrics["edis_weight/log_edis_max"] = log_edis.max().item()
        
        # === Step 2: Centered metrics ===
        metrics["edis_weight/centered_mean"] = centered.mean().item()
        metrics["edis_weight/centered_std"] = centered.std().item()
        metrics["edis_weight/centered_min"] = centered.min().item()
        metrics["edis_weight/centered_max"] = centered.max().item()
        
        # === Step 3: Signed metrics (after correctness flip) ===
        metrics["edis_weight/signed_mean"] = signed.mean().item()
        metrics["edis_weight/signed_std"] = signed.std().item()
        metrics["edis_weight/signed_min"] = signed.min().item()
        metrics["edis_weight/signed_max"] = signed.max().item()
        
        # === Step 4: Softmax parameters ===
        metrics["edis_weight/alpha"] = alpha
        metrics["edis_weight/temperature_k"] = k.item()
        
        # === Step 5: Pre-normalization softmax weights ===
        metrics["edis_weight/softmax_pre_norm_mean"] = softmax_weights.mean().item()
        metrics["edis_weight/softmax_pre_norm_std"] = softmax_weights.std().item()
        metrics["edis_weight/softmax_pre_norm_min"] = softmax_weights.min().item()
        metrics["edis_weight/softmax_pre_norm_max"] = softmax_weights.max().item()
        
        # === Step 6: Group normalization verification ===
        metrics["edis_weight/n_correct"] = n_correct.item()
        metrics["edis_weight/n_wrong"] = n_wrong.item()
        if n_correct > 0:
            metrics["edis_weight/correct_weight_sum"] = group_weights[correct_mask].sum().item()
            metrics["edis_weight/correct_weight_mean"] = group_weights[correct_mask].mean().item()
        if n_wrong > 0:
            metrics["edis_weight/wrong_weight_sum"] = group_weights[wrong_mask].sum().item()
            metrics["edis_weight/wrong_weight_mean"] = group_weights[wrong_mask].mean().item()
        
        # === Final weights statistics ===
        metrics["edis_weight/final_weight_mean"] = group_weights.mean().item()
        metrics["edis_weight/final_weight_std"] = group_weights.std().item()
        metrics["edis_weight/final_weight_min"] = group_weights.min().item()
        metrics["edis_weight/final_weight_max"] = group_weights.max().item()
        
        # === Four quadrant weights ===
        if hc.sum() > 0:
            metrics["edis_weight/weight_high_correct"] = group_weights[hc].mean().item()
            metrics["edis_weight/n_high_correct"] = hc.sum().item()
        if lc.sum() > 0:
            metrics["edis_weight/weight_low_correct"] = group_weights[lc].mean().item()
            metrics["edis_weight/n_low_correct"] = lc.sum().item()
        if hw.sum() > 0:
            metrics["edis_weight/weight_high_wrong"] = group_weights[hw].mean().item()
            metrics["edis_weight/n_high_wrong"] = hw.sum().item()
        if lw.sum() > 0:
            metrics["edis_weight/weight_low_wrong"] = group_weights[lw].mean().item()
            metrics["edis_weight/n_low_wrong"] = lw.sum().item()
        
        # === Target vs non-target ===
        target_mask = hc | lw
        non_target_mask = lc | hw
        if target_mask.sum() > 0 and non_target_mask.sum() > 0:
            target_weight = group_weights[target_mask].mean().item()
            non_target_weight = group_weights[non_target_mask].mean().item()
            metrics["edis_weight/target_weight_mean"] = target_weight
            metrics["edis_weight/non_target_weight_mean"] = non_target_weight
            metrics["edis_weight/target_vs_non_target_ratio"] = target_weight / (non_target_weight + 1e-8)
            metrics["edis_weight/n_target"] = target_mask.sum().item()
            metrics["edis_weight/n_non_target"] = non_target_mask.sum().item()
    
    # === Mixed group statistics (方案4: 只对混合group加权) ===
    metrics["edis_weight/n_groups_total"] = n_groups_total
    metrics["edis_weight/n_groups_mixed"] = n_groups_mixed
    metrics["edis_weight/n_groups_all_correct"] = n_groups_all_correct
    metrics["edis_weight/n_groups_all_wrong"] = n_groups_all_wrong
    metrics["edis_weight/mixed_group_ratio"] = n_groups_mixed / max(n_groups_total, 1)
    metrics["edis_weight/n_samples_weighted"] = n_samples_weighted
    metrics["edis_weight/n_samples_skipped"] = n_samples_skipped
    metrics["edis_weight/samples_weighted_ratio"] = n_samples_weighted / max(n_samples_weighted + n_samples_skipped, 1)
    
    return weights, metrics


def compute_edis_from_entropys(
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5
) -> torch.Tensor:
    """
    Compute EDIS scores from token-level entropy tensors.
    
    This is a batched version for training-time computation.
    
    Formula: EDIS = spike × (1 + variance)
    where spike = 0.5 × (burst_spike + peak_spike)
    
    Args:
        entropys: Token-level entropy tensor. Shape: [batch_size, seq_len]
        response_masks: Mask for valid response tokens. Shape: [batch_size, seq_len]
        burst_threshold: Threshold for burst spike detection. Default 1.36.
        peak_threshold: Threshold for peak-to-trough detection. Default 1.33.
        window: Window size for burst detection. Default 5.
    
    Returns:
        EDIS scores tensor. Shape: [batch_size]
    """
    batch_size = entropys.shape[0]
    edis_scores = []
    
    for i in range(batch_size):
        entropy_seq = entropys[i]
        mask = response_masks[i] if response_masks is not None else None
        
        # Compute spike and variance
        combined_spike, _, _ = compute_hybrid_spike(
            entropy_seq,
            burst_threshold=burst_threshold,
            peak_threshold=peak_threshold,
            window=window,
            mask=mask
        )
        
        entropy_var = compute_entropy_variance(entropy_seq, mask=mask)
        
        # EDIS = spike × (1 + variance)
        edis = compute_edis(spike_count=combined_spike, entropy_variance=entropy_var)
        edis_scores.append(edis)
    
    return torch.tensor(edis_scores, dtype=torch.float32, device=entropys.device)


def apply_edis_weighting_to_advantages(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    token_level_scores: torch.Tensor,
    uids: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5
) -> tuple[torch.Tensor, dict]:
    """
    Apply EDIS-based weighting to advantages using Log-Softmax.
    
    This is the main entry point for Method C (EDIS weighting) in training.
    Uses Log-Softmax with correct/wrong group normalization.
    
    Args:
        advantages: Advantage values. Shape: [batch_size, seq_len]
        entropys: Token-level entropy. Shape: [batch_size, seq_len]
        response_masks: Mask for valid response tokens. Shape: [batch_size, seq_len]
        token_level_scores: Reward scores (correctness). Shape: [batch_size, seq_len]
        uids: Optional array of unique IDs for grouping. Shape: [batch_size]
        alpha: Softmax temperature multiplier.
               控制权重分布的集中/分散程度: k = alpha * std(log_edis)
               - alpha=0.5: 激进 (目标样本 2-5x，少数样本主导)
               - alpha=1.0: 推荐 (目标样本 2-2.5x，平衡)
               - alpha=2.0: 保守 (目标样本 1.5-2x，均匀)
        burst_threshold: Threshold for burst spike detection.
        peak_threshold: Threshold for peak-to-trough detection.
        window: Window size for burst detection.
    
    Returns:
        Tuple of (weighted_advantages, metrics_dict)
    """
    batch_size = advantages.shape[0]
    device = advantages.device
    
    # Step 1: Compute EDIS scores
    edis_scores = compute_edis_from_entropys(
        entropys=entropys,
        response_masks=response_masks,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window
    )
    
    # Step 2: Determine correctness from token_level_scores
    # Use sum > 0 (NOT == 1.0) to handle partial/continuous rewards
    total_reward = token_level_scores.sum(dim=-1)
    is_correct = (total_reward > 0).float()
    
    # Step 3: Group by uid if provided
    group_indices = None
    if uids is not None:
        uid_to_indices = group_indices_by_uid(uids)
        group_indices = list(uid_to_indices.values())
    
    # Step 4: Compute EDIS weights (Log-Softmax + group normalization)
    edis_weights, weight_metrics = compute_edis_weights(
        edis_scores=edis_scores,
        is_correct=is_correct,
        alpha=alpha,
        group_indices=group_indices
    )
    
    # Step 5: Apply weights to advantages
    weights_expanded = edis_weights.unsqueeze(-1).to(device)
    weighted_advantages = advantages * weights_expanded
    
    # Step 6: Normalize weighted_advantages back to original scale
    # (Group normalization already ensures mean ≈ 0, just match scale)
    token_count_per_seq = response_masks.sum(dim=-1)
    orig_sum_per_seq = (advantages * response_masks).sum(dim=-1)
    orig_mean_per_seq = orig_sum_per_seq / (token_count_per_seq + 1e-8)
    orig_loss_scale = orig_mean_per_seq.abs().mean()
    
    new_sum_per_seq = (weighted_advantages * response_masks).sum(dim=-1)
    new_mean_per_seq = new_sum_per_seq / (token_count_per_seq + 1e-8)
    new_loss_scale = new_mean_per_seq.abs().mean()
    
    # Scale to match original loss scale
    if new_loss_scale > 1e-8:
        scale_factor = orig_loss_scale / new_loss_scale
        weighted_advantages = weighted_advantages * scale_factor
    else:
        scale_factor = torch.tensor(1.0, device=device)
    
    # Compute final statistics
    final_sum_per_seq = (weighted_advantages * response_masks).sum(dim=-1)
    final_mean_per_seq = final_sum_per_seq / (token_count_per_seq + 1e-8)
    final_global_mean = final_mean_per_seq.mean()
    final_loss_scale = final_mean_per_seq.abs().mean()
    orig_global_mean = orig_mean_per_seq.mean()
    
    # ============== Sequence Contribution Analysis ==============
    seq_contribution = (weighted_advantages.abs() * response_masks).sum(dim=-1)
    total_contribution = seq_contribution.sum()
    seq_contribution_ratio = seq_contribution / (total_contribution + 1e-8)
    
    sorted_contrib, _ = seq_contribution_ratio.sort(descending=True)
    cumsum_contrib = sorted_contrib.cumsum(dim=0)
    
    top_50pct_count = (cumsum_contrib <= 0.5).sum().item() + 1
    top_80pct_count = (cumsum_contrib <= 0.8).sum().item() + 1
    top_90pct_count = (cumsum_contrib <= 0.9).sum().item() + 1
    
    n = len(seq_contribution_ratio)
    sorted_contrib_for_gini = seq_contribution_ratio.sort()[0]
    cum_contrib = sorted_contrib_for_gini.cumsum(dim=0)
    gini = 1 - 2 * cum_contrib.sum() / (n * seq_contribution_ratio.sum() + 1e-8)
    
    raw_contribution = (advantages.abs() * response_masks).sum(dim=-1)
    raw_total = raw_contribution.sum()
    raw_ratio = raw_contribution / (raw_total + 1e-8)
    sorted_raw, _ = raw_ratio.sort(descending=True)
    cumsum_raw = sorted_raw.cumsum(dim=0)
    raw_top_50pct = (cumsum_raw <= 0.5).sum().item() + 1
    raw_top_90pct = (cumsum_raw <= 0.9).sum().item() + 1
    # ============== End Contribution Analysis ==============
    
    # Collect metrics
    metrics = {
        # EDIS statistics
        "edis_weight/edis_mean": edis_scores.mean().item(),
        "edis_weight/edis_std": edis_scores.std().item(),
        # Weight statistics
        "edis_weight/weight_mean": edis_weights.mean().item(),
        "edis_weight/weight_std": edis_weights.std().item(),
        "edis_weight/weight_min": edis_weights.min().item(),
        "edis_weight/weight_max": edis_weights.max().item(),
        # Scale normalization
        "edis_weight/scale_factor": scale_factor.item() if isinstance(scale_factor, torch.Tensor) else scale_factor,
        # Advantage statistics
        "edis_weight/orig_adv_global_mean": orig_global_mean.item(),
        "edis_weight/orig_adv_loss_scale": orig_loss_scale.item(),
        "edis_weight/final_adv_global_mean": final_global_mean.item(),
        "edis_weight/final_adv_loss_scale": final_loss_scale.item(),
        # Correctness statistics
        "edis_weight/correct_ratio": is_correct.mean().item(),
        "edis_weight/total_reward_mean": total_reward.mean().item(),
        # Contribution analysis
        "edis_weight/contrib_gini": gini.item(),
        "edis_weight/contrib_top1_ratio": sorted_contrib[0].item(),
        "edis_weight/contrib_top10pct_ratio": cumsum_contrib[max(0, n//10-1)].item() if n >= 10 else 1.0,
        "edis_weight/contrib_seqs_for_50pct": top_50pct_count,
        "edis_weight/contrib_seqs_for_80pct": top_80pct_count,
        "edis_weight/contrib_seqs_for_90pct": top_90pct_count,
        "edis_weight/raw_contrib_seqs_for_50pct": raw_top_50pct,
        "edis_weight/raw_contrib_seqs_for_90pct": raw_top_90pct,
    }
    
    # Merge weight_metrics from compute_edis_weights
    metrics.update(weight_metrics)
    
    # Metrics by correctness
    if is_correct.sum() > 0:
        metrics["edis_weight/edis_correct_mean"] = edis_scores[is_correct > 0.5].mean().item()
        metrics["edis_weight/weight_correct_mean"] = edis_weights[is_correct > 0.5].mean().item()
    if (1 - is_correct).sum() > 0:
        metrics["edis_weight/edis_wrong_mean"] = edis_scores[is_correct < 0.5].mean().item()
        metrics["edis_weight/weight_wrong_mean"] = edis_weights[is_correct < 0.5].mean().item()
    
    return weighted_advantages, metrics


# Backward compatibility alias
apply_focal_weighting_to_advantages = apply_edis_weighting_to_advantages
compute_focal_weights = compute_edis_weights


# ============================================================================
# EDIS-based Sample Selection for Training (Method F only)
# ============================================================================

def select_extreme_edis_samples(
    edis_scores: torch.Tensor,
    uids: np.ndarray,
    target_n: int,
    multiplier: float = 2.0,
    scores: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict]:
    """
    Select samples by correctness + EDIS for training (Method F).
    
    方案F：按正确性+EDIS选择 ⭐
      - 正确样本中选EDIS最低的（确定正确 → 强化）
      - 错误样本中选EDIS最高的（不确定错误 → 惩罚）
      - 交替选取直到满n个，保持正确:错误尽量平衡
      - 实验结果: 优于baseline +0.84%
    
    Args:
        edis_scores: EDIS scores for all samples. Shape: [batch_size * gen_n]
        uids: Unique IDs for grouping (same uid = same prompt). Shape: [batch_size * gen_n]
        target_n: Target number of samples per group (typically 8).
        multiplier: How many times more samples were generated (can be float, e.g., 1.5).
        scores: Sample scores (>0 correct, <=0 wrong). Shape: [batch_size * gen_n].
                REQUIRED - must be provided for Method F.
    
    Returns:
        selected_mask: Boolean mask for selected samples. Shape: [batch_size * multiplier]
        metrics: Dictionary with selection statistics.
    
    Raises:
        ValueError: If scores is None (Method D is deprecated and removed).
    """
    # Method D is deprecated and removed - scores must be provided
    if scores is None:
        raise ValueError(
            "scores must be provided for EDIS filtering (Method F). "
            "Method D (EDIS-only) has been deprecated and removed."
        )
    
    device = edis_scores.device
    batch_size = len(edis_scores)
    
    # Group by uid
    unique_uids = np.unique(uids)
    selected_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Metrics counters
    n_correct_selected = 0
    n_wrong_selected = 0
    n_groups = 0
    
    for uid in unique_uids:
        # Find samples for this group
        group_mask = uids == uid
        group_indices = torch.where(torch.tensor(group_mask, device=device))[0]
        group_edis = edis_scores[group_indices]
        
        if len(group_indices) < target_n:
            # Not enough samples, select all
            selected_mask[group_indices] = True
            continue
        
        n_groups += 1
        
        # ========== 方案F: 按正确性+EDIS选择 ==========
        group_scores = scores[group_indices]
        
        # 分离正确和错误样本
        correct_mask = group_scores > 0
        wrong_mask = group_scores <= 0
        
        correct_local_indices = torch.where(correct_mask)[0]  # indices within group
        wrong_local_indices = torch.where(wrong_mask)[0]
        
        # 正确样本按EDIS升序（选最低的）
        if len(correct_local_indices) > 0:
            correct_edis = group_edis[correct_local_indices]
            correct_sorted = correct_local_indices[torch.argsort(correct_edis)]  # 升序
        else:
            correct_sorted = torch.tensor([], dtype=torch.long, device=device)
        
        # 错误样本按EDIS降序（选最高的）
        if len(wrong_local_indices) > 0:
            wrong_edis = group_edis[wrong_local_indices]
            wrong_sorted = wrong_local_indices[torch.argsort(wrong_edis, descending=True)]  # 降序
        else:
            wrong_sorted = torch.tensor([], dtype=torch.long, device=device)
        
        # 交替选取，直到满target_n个
        selected_local = []
        i_correct, i_wrong = 0, 0
        
        while len(selected_local) < target_n:
            # 先尝试选一个正确的
            if i_correct < len(correct_sorted):
                selected_local.append(correct_sorted[i_correct].item())
                i_correct += 1
                n_correct_selected += 1
            
            if len(selected_local) >= target_n:
                break
            
            # 再尝试选一个错误的
            if i_wrong < len(wrong_sorted):
                selected_local.append(wrong_sorted[i_wrong].item())
                i_wrong += 1
                n_wrong_selected += 1
            
            # 如果两边都用完了但还没满，继续从剩余的一方选
            if i_correct >= len(correct_sorted) and i_wrong >= len(wrong_sorted):
                break
        
        # 转换回全局索引并标记
        for local_idx in selected_local:
            global_idx = group_indices[local_idx]
            selected_mask[global_idx] = True
    
    n_selected = selected_mask.sum().item()
    n_total = batch_size
    
    metrics = {
        "train_edis_filter/n_groups": n_groups,
        "train_edis_filter/n_correct_selected": n_correct_selected,
        "train_edis_filter/n_wrong_selected": n_wrong_selected,
        "train_edis_filter/n_selected": n_selected,
        "train_edis_filter/n_total": n_total,
        "train_edis_filter/selection_ratio": n_selected / max(n_total, 1),
        "train_edis_filter/multiplier": multiplier,
        "train_edis_filter/target_n_per_group": target_n,
        "train_edis_filter/method": "F",
    }
    
    return selected_mask, metrics


def filter_batch_by_edis(
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    uids: np.ndarray,
    target_n: int,
    multiplier: float = 2.0,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
    scores: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict]:
    """
    Compute EDIS and return selection mask for samples (Method F).
    
    This is the main entry point for EDIS filtering in training.
    
    方案F：按正确性+EDIS选择 ⭐
      - From correct samples: select lowest EDIS (most confident correct → reinforce)
      - From wrong samples: select highest EDIS (most uncertain wrong → penalize)
      - Alternates between correct and wrong to maintain balance
      - Experimental result: +0.84% vs baseline
    
    Args:
        entropys: Token-level entropy. Shape: [batch_size, seq_len]
        response_masks: Mask for valid response tokens. Shape: [batch_size, seq_len]
        uids: Unique IDs for grouping. Shape: [batch_size]
        target_n: Target number of samples per group.
        multiplier: How many times more samples were generated (can be float, e.g., 1.5).
        burst_threshold: Threshold for burst spike detection.
        peak_threshold: Threshold for peak-to-trough detection.
        window: Window size for burst detection.
        scores: Sample scores for correctness (>0 correct, <=0 wrong). 
                REQUIRED for Method F.
    
    Returns:
        selected_mask: Boolean mask for selected samples. Shape: [batch_size]
        metrics: Dictionary with filtering statistics.
    
    Raises:
        ValueError: If scores is None (must be provided for Method F).
    """
    # Step 1: Compute EDIS for all samples
    edis_scores = compute_edis_from_entropys(
        entropys=entropys,
        response_masks=response_masks,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window
    )
    
    # Step 2: Select samples using Method F (correctness + EDIS)
    selected_mask, metrics = select_extreme_edis_samples(
        edis_scores=edis_scores,
        uids=uids,
        target_n=target_n,
        multiplier=multiplier,
        scores=scores
    )
    
    # Add EDIS statistics to metrics
    metrics["train_edis_filter/edis_mean"] = edis_scores.mean().item()
    metrics["train_edis_filter/edis_std"] = edis_scores.std().item()
    metrics["train_edis_filter/edis_min"] = edis_scores.min().item()
    metrics["train_edis_filter/edis_max"] = edis_scores.max().item()
    
    # EDIS of selected vs discarded
    if selected_mask.sum() > 0:
        metrics["train_edis_filter/selected_edis_mean"] = edis_scores[selected_mask].mean().item()
    if (~selected_mask).sum() > 0:
        metrics["train_edis_filter/discarded_edis_mean"] = edis_scores[~selected_mask].mean().item()
    
    return selected_mask, metrics


# ============================================================================
# Calibration Reward for Training (Method B)
# ============================================================================

def compute_entropy_reward(
    edis_scores: torch.Tensor,
    coef: float = 0.1,
    k: float = 1.0,
    group_indices: Optional[list[list[int]]] = None
) -> tuple[torch.Tensor, dict]:
    """
    Compute Entropy Reward based on EDIS (stability reward).
    
    Core idea: Reward stable reasoning, penalize unstable reasoning.
        - Low EDIS (stable) → Reward (+coef)
        - High EDIS (unstable) → Penalty (-coef)
        - Median EDIS → No effect (0)
    
    Formula (Median-Split + tanh):
        1. norm = Median-Split(EDIS) → [-1, 1], center=0
        2. entropy_reward = -tanh(k * norm) * coef
           - tanh 保证输出范围在 (-1, 1)
           - 乘以 coef 后范围变为 (-coef, +coef)
           - Low EDIS → norm negative → reward positive
           - High EDIS → norm positive → reward negative
    
    Args:
        edis_scores: EDIS scores for each sample. Shape: [batch_size]
        coef: Reward coefficient. 0=disabled, 0.1=±0.1 range. Default 0.1.
        k: Curve steepness for tanh. Controls transition sharpness.
           - k=1: standard (default)
           - k<1: flatter curve
           - k>1: steeper curve (faster saturation)
        group_indices: List of index lists for each group. If None, treat all as one group.
    
    Returns:
        Tuple of (entropy_reward, metrics_dict)
        - entropy_reward: Shape [batch_size], range (-coef, +coef)
    """
    # Early return if disabled
    if coef == 0:
        batch_size = len(edis_scores) if isinstance(edis_scores, (list, np.ndarray)) else edis_scores.shape[0]
        device = edis_scores.device if isinstance(edis_scores, torch.Tensor) else 'cpu'
        return torch.zeros(batch_size, dtype=torch.float32, device=device), {}
    
    # Convert to tensors
    if isinstance(edis_scores, list):
        edis_scores = torch.tensor(edis_scores, dtype=torch.float32)
    elif isinstance(edis_scores, np.ndarray):
        edis_scores = torch.from_numpy(edis_scores).float()
    
    device = edis_scores.device if isinstance(edis_scores, torch.Tensor) else 'cpu'
    edis_scores = edis_scores.to(device)
    
    batch_size = len(edis_scores)
    entropy_reward = torch.zeros(batch_size, dtype=torch.float32, device=device)
    norm_all = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # If no group_indices, treat all samples as one group
    if group_indices is None:
        group_indices = [list(range(batch_size))]
    
    for indices in group_indices:
        if len(indices) == 0:
            continue
        
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        group_edis = edis_scores[indices_tensor]
        
        # Step 1: Median-Split normalization (guarantees center=0)
        edis_min = group_edis.min()
        edis_max = group_edis.max()
        edis_median = group_edis.median()
        
        # Handle edge cases
        if edis_max - edis_min < 1e-8:
            # All EDIS are the same, no discriminative power
            norm_all[indices_tensor] = 0.0
            entropy_reward[indices_tensor] = 0.0
            continue
        
        # Median-Split: map median to 0.5, min to 0, max to 1
        norm_01 = torch.zeros_like(group_edis)
        below_median = group_edis < edis_median
        above_median = ~below_median
        
        if below_median.sum() > 0:
            denom = edis_median - edis_min + 1e-8
            norm_01[below_median] = 0.5 * (group_edis[below_median] - edis_min) / denom
        
        if above_median.sum() > 0:
            denom = edis_max - edis_median + 1e-8
            norm_01[above_median] = 0.5 + 0.5 * (group_edis[above_median] - edis_median) / denom
        
        # Map to [-1, 1], center=0
        norm_11 = 2.0 * norm_01 - 1.0
        
        # Step 2: Entropy reward with tanh
        # tanh(k * norm) maps [-1, 1] → (-1, 1) with controlled steepness
        # Low EDIS → norm negative → reward positive (reward stability)
        # High EDIS → norm positive → reward negative (penalize instability)
        group_reward = -torch.tanh(k * norm_11) * coef
        
        # Assign back
        norm_all[indices_tensor] = norm_11
        entropy_reward[indices_tensor] = group_reward
    
    # Collect metrics
    metrics = {
        "entropy_reward/norm_mean": norm_all.mean().item(),
        "entropy_reward/norm_std": norm_all.std().item(),
        "entropy_reward/reward_mean": entropy_reward.mean().item(),
        "entropy_reward/reward_std": entropy_reward.std().item(),
        "entropy_reward/reward_min": entropy_reward.min().item(),
        "entropy_reward/reward_max": entropy_reward.max().item(),
    }
    
    return entropy_reward, metrics


def apply_entropy_reward(
    rewards: torch.Tensor,
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    uids: Optional[np.ndarray] = None,
    coef: float = 0.1,
    k: float = 1.0,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5
) -> tuple[torch.Tensor, dict]:
    """
    Apply entropy reward to rewards based on EDIS (stability reward).
    
    This is the main entry point for Method B (Entropy Reward) in training.
    
    Core idea:
        - Low EDIS (stable reasoning) → Extra reward (+coef)
        - High EDIS (unstable reasoning) → Extra penalty (-coef)
        - Independent of correctness!
    
    Args:
        rewards: Token-level rewards. Shape: [batch_size, seq_len]
        entropys: Token-level entropy. Shape: [batch_size, seq_len]
        response_masks: Mask for valid response tokens. Shape: [batch_size, seq_len]
        uids: Optional array of unique IDs for grouping. Shape: [batch_size]
        coef: Entropy reward coefficient. 0=disabled, 0.1=±0.1 range. Default 0.1.
        k: Curve steepness for tanh. k=1 standard, k<1 flatter, k>1 steeper. Default 1.0.
        burst_threshold: Threshold for burst spike detection.
        peak_threshold: Threshold for peak-to-trough detection.
        window: Window size for burst detection.
    
    Returns:
        Tuple of (adjusted_rewards, metrics_dict)
        - adjusted_rewards: rewards + entropy_reward, Shape: [batch_size, seq_len]
    """
    batch_size = rewards.shape[0]
    device = rewards.device
    
    # Step 1: Compute EDIS scores
    edis_scores = compute_edis_from_entropys(
        entropys=entropys,
        response_masks=response_masks,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window
    )
    
    # Step 2: Group by uid if provided
    group_indices = None
    if uids is not None:
        uid_to_indices = group_indices_by_uid(uids)
        group_indices = list(uid_to_indices.values())
    
    # Step 3: Compute entropy reward (sample-level)
    # Low EDIS → reward, High EDIS → penalty
    entropy_reward, reward_metrics = compute_entropy_reward(
        edis_scores=edis_scores,
        coef=coef,
        k=k,
        group_indices=group_indices
    )
    
    # Step 4: Apply entropy reward to rewards
    # Add to the last valid token of each sample
    adjusted_rewards = rewards.clone()
    
    for i in range(batch_size):
        # Find last valid token
        valid_positions = response_masks[i].nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            last_pos = valid_positions[-1]
            adjusted_rewards[i, last_pos] = adjusted_rewards[i, last_pos] + entropy_reward[i]
    
    # Add EDIS metrics
    metrics = {
        "entropy_reward/edis_mean": edis_scores.mean().item(),
        "entropy_reward/edis_std": edis_scores.std().item(),
    }
    metrics.update(reward_metrics)
    
    return adjusted_rewards, metrics
