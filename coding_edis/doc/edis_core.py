#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDIS (Entropy Dynamics Instability Score) 核心计算模块

这是一个独立的、轻量级的 EDIS 计算模块，不依赖 verl 框架，
可以方便地在其他项目中复用。

核心公式:
    EDIS = spike × (1 + variance)
    
    其中:
    - spike = 0.5 × (burst_spike + peak_trough_spike)
    - variance = Var(entropy_sequence)

设计原理:
    - 错误响应的熵突变次数是正确响应的 1.8-2.0 倍
    - 低 EDIS → 推理稳定、流畅
    - 高 EDIS → 推理不稳定、多次犹豫

使用示例:
    >>> from edis_core import compute_edis_from_sequence
    >>> entropy_seq = [0.5, 0.8, 1.2, 1.8, 1.5, 2.1, 0.9]
    >>> edis, metrics = compute_edis_from_sequence(entropy_seq)
    >>> print(f"EDIS: {edis:.4f}")
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np

__version__ = "1.0.0"
__all__ = [
    # 核心函数
    "compute_edis",
    "compute_edis_from_sequence",
    # Spike 检测
    "compute_burst_spike",
    "compute_peak_spike",
    "compute_hybrid_spike",
    # 辅助函数
    "compute_entropy_variance",
    "compute_mean_entropy",
    # 批量处理
    "compute_edis_batch",
]


# =============================================================================
# 默认参数 (经过 1700+ 种组合测试优化)
# =============================================================================

DEFAULT_BURST_THRESHOLD = 1.36  # 突发检测阈值
DEFAULT_PEAK_THRESHOLD = 1.33   # 峰谷检测阈值
DEFAULT_WINDOW = 5              # 滑动窗口大小


# =============================================================================
# 1. 突发 Spike 检测 (Burst Spike) - 检测"持续恶化"
# =============================================================================

def compute_burst_spike(
    entropy_sequence: Union[np.ndarray, List[float]],
    threshold: float = DEFAULT_BURST_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[np.ndarray] = None
) -> int:
    """
    计算突发 Spike：检测窗口内累计熵增长超过阈值的情况。
    
    原理: 在滑动窗口(5个token)内，如果熵的累计增长超过阈值，
          说明模型在"持续犹豫"。
    
    公式: burst_spike = Σ I(entropy[i + window] - entropy[i] > threshold)
    
    形象比喻: 连续5天血压是否持续上涨超过阈值
    
    捕捉的模式:
        ✓ [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5]  # 缓慢但持续上升
        ✗ [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]              # 反复跳跃但无趋势
    
    Args:
        entropy_sequence: 熵序列，每个位置的熵值
        threshold: 累计增长阈值，默认 1.36
        window: 滑动窗口大小，默认 5
        mask: 可选的掩码，1=有效，0=填充
    
    Returns:
        突发 spike 次数
    
    Example:
        >>> seq = [1.0, 1.2, 1.4, 1.6, 1.8, 2.6]
        >>> compute_burst_spike(seq, threshold=1.36, window=5)
        1  # seq[5] - seq[0] = 2.6 - 1.0 = 1.6 > 1.36 → spike!
    """
    # 转换为 numpy 数组
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    # 应用掩码
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < window + 1:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < window + 1:
        return 0
    
    # 计算窗口首尾之差超过阈值的次数
    count = 0
    for i in range(len(entropy_sequence) - window):
        cumulative_increase = entropy_sequence[i + window] - entropy_sequence[i]
        if cumulative_increase > threshold:
            count += 1
    
    return count


# =============================================================================
# 2. 峰谷 Spike 检测 (Peak-to-Trough Spike) - 检测"V型反弹"
# =============================================================================

def compute_peak_spike(
    entropy_sequence: Union[np.ndarray, List[float]],
    threshold: float = DEFAULT_PEAK_THRESHOLD,
    mask: Optional[np.ndarray] = None
) -> int:
    """
    计算峰谷 Spike：检测从历史最低点反弹超过阈值的情况。
    
    原理: 检测熵从历史最低点反弹超过阈值的情况，
          捕捉"以为懂了但又糊涂了"的模式。
    
    公式: peak_trough_spike = Σ I(entropy[i] - min(entropy[:i]) > threshold)
    
    形象比喻: 血压从最低点反弹超过30mmHg
    
    捕捉的模式:
        ✓ [2.0, 1.0, 0.5, 1.0, 2.0]  # V型反弹：从低点大幅上升
        ✓ [0.5, 0.3, 0.8, 0.2, 1.6]  # 多次反弹：反复"以为好了又差了"
        ✗ [0.5, 0.6, 0.7, 0.8, 0.9]  # 单调上升但无明显低点反弹
    
    Args:
        entropy_sequence: 熵序列，每个位置的熵值
        threshold: 反弹阈值，默认 1.33
        mask: 可选的掩码，1=有效，0=填充
    
    Returns:
        峰谷 spike 次数
    
    Example:
        >>> seq = [2.0, 1.5, 1.0, 0.5, 0.8, 1.5, 2.0]
        >>> compute_peak_spike(seq, threshold=1.33)
        1  # 位置6: min=0.5, current=2.0, rebound=1.5 > 1.33 → spike!
    """
    # 转换为 numpy 数组
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    # 应用掩码
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < 2:
        return 0
    
    # 跟踪历史最低点，计算反弹次数
    count = 0
    historical_min = entropy_sequence[0]
    
    for i in range(1, len(entropy_sequence)):
        current = entropy_sequence[i]
        # 检查当前值是否从历史最低点反弹超过阈值
        rebound = current - historical_min
        if rebound > threshold:
            count += 1
        # 更新历史最低点
        if current < historical_min:
            historical_min = current
    
    return count


# =============================================================================
# 3. 混合 Spike 计算 (Hybrid Spike) - 组合两种方法
# =============================================================================

def compute_hybrid_spike(
    entropy_sequence: Union[np.ndarray, List[float]],
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[np.ndarray] = None
) -> Tuple[float, int, int]:
    """
    计算混合 Spike：突发 + 峰谷的加权平均。
    
    公式: spike = 0.5 × (burst_spike + peak_trough_spike)
    
    这是我们的最终方法！经过 1700+ 种组合测试验证：
    - 准确率：52.19% (原始 32.3% → 提升 19.9 个百分点)
    - 区分度：1.84x (错误/正确 EDIS 比值)
    
    加权平均的优势：
        1. 灵活性：两种模式贡献可调节
        2. 平滑性：不会因单一极端值失真
        3. 互补性：burst 捕捉"持续恶化"，peak-trough 捕捉"V型反弹"
    
    Args:
        entropy_sequence: 熵序列
        burst_threshold: 突发检测阈值，默认 1.36
        peak_threshold: 峰谷检测阈值，默认 1.33
        window: 滑动窗口大小，默认 5
        mask: 可选的掩码
    
    Returns:
        (combined_spike, burst_spike, peak_spike) 三元组
        - combined_spike: 加权平均值，用于 EDIS 计算
        - burst_spike: 突发 spike 原始计数
        - peak_spike: 峰谷 spike 原始计数
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
    
    # 加权平均：可解释为"熵激变平均次数"
    combined_spike = 0.5 * (burst_spike + peak_spike)
    
    return combined_spike, burst_spike, peak_spike


# =============================================================================
# 4. 熵方差计算
# =============================================================================

def compute_entropy_variance(
    entropy_sequence: Union[np.ndarray, List[float]],
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算熵序列的方差。
    
    方差衡量熵值的波动程度，高方差表示熵值变化剧烈。
    
    Args:
        entropy_sequence: 熵序列
        mask: 可选的掩码
    
    Returns:
        熵方差值
    """
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < 2:
        return 0.0
    
    return float(np.var(entropy_sequence))


# =============================================================================
# 5. 平均熵计算
# =============================================================================

def compute_mean_entropy(
    entropy_sequence: Union[np.ndarray, List[float]],
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算熵序列的平均值。
    
    Args:
        entropy_sequence: 熵序列
        mask: 可选的掩码
    
    Returns:
        平均熵值
    """
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 1:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]
    
    if len(entropy_sequence) < 1:
        return 0.0
    
    return float(np.mean(entropy_sequence))


# =============================================================================
# 6. EDIS 核心计算
# =============================================================================

def compute_edis(
    spike_count: float,
    entropy_variance: float
) -> float:
    """
    计算 EDIS (Entropy Dynamics Instability Score)。
    
    这是 EDIS 的核心公式！
    
    公式:
        EDIS = spike × (1 + variance)
    
    设计原理:
        - spike_count: 熵突变次数（burst + peak 混合方法）
          不按长度归一化！因为错误响应往往更长，有更多 spike
        - variance: 熵波动幅度
        - 乘法组合: spike 和 variance 相互放大不稳定性
    
    含义:
        - 低 EDIS → 推理稳定、流畅 → 更可能正确
        - 高 EDIS → 推理不稳定、多次犹豫 → 更可能错误
    
    实验性能 (hybrid burst + peak-to-trough 方法):
        - 区分度: 1.84x (wrong/correct EDIS)
        - 筛选准确率: 52.2% (vs 32.3% baseline)
        - 最优参数: burst=1.36 (window=5), peak=1.33
    
    Args:
        spike_count: spike 分数 (float，来自 hybrid 方法)
        entropy_variance: 熵方差
    
    Returns:
        EDIS 分数 (越低越好，表示生成越稳定)
    """
    edis = spike_count * (1 + entropy_variance)
    return float(edis)


# =============================================================================
# 7. 从熵序列直接计算 EDIS (一站式接口)
# =============================================================================

def compute_edis_from_sequence(
    entropy_sequence: Union[np.ndarray, List[float]],
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[np.ndarray] = None,
    return_details: bool = True
) -> Union[float, Tuple[float, Dict]]:
    """
    从熵序列直接计算 EDIS（一站式接口）。
    
    这是最常用的函数，封装了完整的 EDIS 计算流程。
    
    计算流程:
        1. 计算 burst spike (持续恶化检测)
        2. 计算 peak spike (V型反弹检测)
        3. 计算混合 spike = 0.5 × (burst + peak)
        4. 计算熵方差
        5. EDIS = spike × (1 + variance)
    
    Args:
        entropy_sequence: 熵序列
        burst_threshold: 突发检测阈值，默认 1.36
        peak_threshold: 峰谷检测阈值，默认 1.33
        window: 滑动窗口大小，默认 5
        mask: 可选的掩码
        return_details: 是否返回详细指标，默认 True
    
    Returns:
        如果 return_details=False: 只返回 EDIS 分数
        如果 return_details=True: 返回 (edis, metrics_dict) 元组
            metrics_dict 包含:
                - spike: 混合 spike 值
                - burst_spike: 突发 spike 计数
                - peak_spike: 峰谷 spike 计数
                - variance: 熵方差
                - mean_entropy: 平均熵
                - seq_length: 序列长度
    
    Example:
        >>> entropy_seq = [0.5, 0.8, 1.2, 1.8, 1.5, 2.1, 0.9, 1.5]
        >>> edis, metrics = compute_edis_from_sequence(entropy_seq)
        >>> print(f"EDIS: {edis:.4f}")
        >>> print(f"Spike: {metrics['spike']}, Variance: {metrics['variance']:.4f}")
    """
    # 计算各项指标
    combined_spike, burst_spike, peak_spike = compute_hybrid_spike(
        entropy_sequence,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window,
        mask=mask
    )
    
    variance = compute_entropy_variance(entropy_sequence, mask=mask)
    mean_entropy = compute_mean_entropy(entropy_sequence, mask=mask)
    
    # 计算 EDIS
    edis = compute_edis(spike_count=combined_spike, entropy_variance=variance)
    
    if not return_details:
        return edis
    
    # 返回详细指标
    if isinstance(entropy_sequence, list):
        seq_length = len(entropy_sequence)
    else:
        seq_length = len(entropy_sequence)
        if mask is not None:
            seq_length = int(np.sum(mask > 0))
    
    metrics = {
        "edis": edis,
        "spike": combined_spike,
        "burst_spike": burst_spike,
        "peak_spike": peak_spike,
        "variance": variance,
        "mean_entropy": mean_entropy,
        "seq_length": seq_length,
    }
    
    return edis, metrics


# =============================================================================
# 8. 批量计算 EDIS
# =============================================================================

def compute_edis_batch(
    entropy_sequences: List[Union[np.ndarray, List[float]]],
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    masks: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    批量计算多个熵序列的 EDIS。
    
    Args:
        entropy_sequences: 熵序列列表
        burst_threshold: 突发检测阈值
        peak_threshold: 峰谷检测阈值
        window: 滑动窗口大小
        masks: 可选的掩码列表，与 entropy_sequences 一一对应
    
    Returns:
        (edis_array, metrics_list) 元组
        - edis_array: EDIS 分数数组，shape [batch_size]
        - metrics_list: 每个序列的详细指标列表
    
    Example:
        >>> sequences = [[0.5, 0.8, 1.2, 1.8], [0.3, 0.4, 0.5, 0.6]]
        >>> edis_scores, all_metrics = compute_edis_batch(sequences)
        >>> print(f"EDIS scores: {edis_scores}")
    """
    batch_size = len(entropy_sequences)
    edis_scores = np.zeros(batch_size, dtype=np.float32)
    all_metrics = []
    
    for i in range(batch_size):
        mask = masks[i] if masks is not None else None
        edis, metrics = compute_edis_from_sequence(
            entropy_sequences[i],
            burst_threshold=burst_threshold,
            peak_threshold=peak_threshold,
            window=window,
            mask=mask,
            return_details=True
        )
        edis_scores[i] = edis
        all_metrics.append(metrics)
    
    return edis_scores, all_metrics


# =============================================================================
# 命令行测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EDIS (Entropy Dynamics Instability Score) 计算示例")
    print("=" * 60)
    
    # 示例 1: 稳定的熵序列（低 EDIS）
    stable_seq = [0.5, 0.6, 0.55, 0.58, 0.52, 0.54, 0.56, 0.53]
    edis_stable, metrics_stable = compute_edis_from_sequence(stable_seq)
    
    print("\n[示例1] 稳定的熵序列:")
    print(f"  序列: {stable_seq}")
    print(f"  EDIS: {edis_stable:.4f}")
    print(f"  Spike: {metrics_stable['spike']}, Variance: {metrics_stable['variance']:.6f}")
    
    # 示例 2: 不稳定的熵序列（高 EDIS）
    unstable_seq = [0.5, 1.2, 0.8, 2.1, 0.6, 2.5, 1.0, 2.8]
    edis_unstable, metrics_unstable = compute_edis_from_sequence(unstable_seq)
    
    print("\n[示例2] 不稳定的熵序列:")
    print(f"  序列: {unstable_seq}")
    print(f"  EDIS: {edis_unstable:.4f}")
    print(f"  Spike: {metrics_unstable['spike']}, Variance: {metrics_unstable['variance']:.6f}")
    
    # 示例 3: 持续恶化模式（触发 burst spike）
    burst_seq = [0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0]
    edis_burst, metrics_burst = compute_edis_from_sequence(burst_seq)
    
    print("\n[示例3] 持续恶化模式 (触发 burst spike):")
    print(f"  序列: {burst_seq}")
    print(f"  EDIS: {edis_burst:.4f}")
    print(f"  Burst spike: {metrics_burst['burst_spike']}, Peak spike: {metrics_burst['peak_spike']}")
    
    # 示例 4: V型反弹模式（触发 peak spike）
    peak_seq = [2.0, 1.5, 1.0, 0.5, 0.8, 1.5, 2.0, 2.5]
    edis_peak, metrics_peak = compute_edis_from_sequence(peak_seq)
    
    print("\n[示例4] V型反弹模式 (触发 peak spike):")
    print(f"  序列: {peak_seq}")
    print(f"  EDIS: {edis_peak:.4f}")
    print(f"  Burst spike: {metrics_peak['burst_spike']}, Peak spike: {metrics_peak['peak_spike']}")
    
    # 对比
    print("\n" + "=" * 60)
    print("EDIS 对比:")
    print("-" * 60)
    print(f"  稳定序列 EDIS:     {edis_stable:.4f} (低 → 更可能正确)")
    print(f"  不稳定序列 EDIS:   {edis_unstable:.4f} (高 → 更可能错误)")
    print(f"  持续恶化 EDIS:     {edis_burst:.4f}")
    print(f"  V型反弹 EDIS:      {edis_peak:.4f}")
    print("=" * 60)
