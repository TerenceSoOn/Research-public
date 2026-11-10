#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDIS (Entropy Dynamics Instability Score) core implementation.
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np

__version__ = "1.0.0"
__all__ = [
    "compute_edis",
    "compute_edis_from_sequence",
    "compute_burst_spike",
    "compute_peak_spike",
    "compute_hybrid_spike",
    "compute_entropy_variance",
    "compute_mean_entropy",
    "compute_edis_batch",
]

DEFAULT_BURST_THRESHOLD = 1.36
DEFAULT_PEAK_THRESHOLD = 1.33
DEFAULT_WINDOW = 5

ArrayLike = Union[np.ndarray, List[float]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    return np.array(x) if isinstance(x, list) else x


def compute_burst_spike(
    entropy_sequence: ArrayLike,
    threshold: float = DEFAULT_BURST_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[ArrayLike] = None,
) -> int:
    entropy_sequence = _to_numpy(entropy_sequence)
    if mask is not None:
        mask = _to_numpy(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < window + 1:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]

    if len(entropy_sequence) < window + 1:
        return 0

    count = 0
    for i in range(len(entropy_sequence) - window):
        cumulative_increase = entropy_sequence[i + window] - entropy_sequence[i]
        if cumulative_increase > threshold:
            count += 1

    return count


def compute_peak_spike(
    entropy_sequence: ArrayLike,
    threshold: float = DEFAULT_PEAK_THRESHOLD,
    mask: Optional[ArrayLike] = None,
) -> int:
    entropy_sequence = _to_numpy(entropy_sequence)
    if mask is not None:
        mask = _to_numpy(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]

    if len(entropy_sequence) < 2:
        return 0

    count = 0
    historical_min = entropy_sequence[0]

    for i in range(1, len(entropy_sequence)):
        current = entropy_sequence[i]
        rebound = current - historical_min
        if rebound > threshold:
            count += 1
        if current < historical_min:
            historical_min = current

    return count


def compute_hybrid_spike(
    entropy_sequence: ArrayLike,
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[ArrayLike] = None,
) -> Tuple[float, int, int]:
    burst_spike = compute_burst_spike(
        entropy_sequence,
        threshold=burst_threshold,
        window=window,
        mask=mask,
    )

    peak_spike = compute_peak_spike(
        entropy_sequence,
        threshold=peak_threshold,
        mask=mask,
    )

    combined_spike = 0.5 * (burst_spike + peak_spike)

    return combined_spike, burst_spike, peak_spike


def compute_entropy_variance(
    entropy_sequence: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> float:
    entropy_sequence = _to_numpy(entropy_sequence)
    if mask is not None:
        mask = _to_numpy(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]

    if len(entropy_sequence) < 2:
        return 0.0

    return float(np.var(entropy_sequence))


def compute_mean_entropy(
    entropy_sequence: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> float:
    entropy_sequence = _to_numpy(entropy_sequence)
    if mask is not None:
        mask = _to_numpy(mask)
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 1:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]

    if len(entropy_sequence) < 1:
        return 0.0

    return float(np.mean(entropy_sequence))


def compute_edis(spike_count: float, entropy_variance: float) -> float:
    return float(spike_count * (1 + entropy_variance))


def compute_edis_from_sequence(
    entropy_sequence: ArrayLike,
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    mask: Optional[ArrayLike] = None,
    return_details: bool = True,
) -> Union[float, Tuple[float, Dict]]:
    combined_spike, burst_spike, peak_spike = compute_hybrid_spike(
        entropy_sequence,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window,
        mask=mask,
    )

    variance = compute_entropy_variance(entropy_sequence, mask=mask)
    edis = compute_edis(combined_spike, variance)

    if not return_details:
        return edis

    metrics = {
        "edis": edis,
        "spike": combined_spike,
        "burst_spike": burst_spike,
        "peak_spike": peak_spike,
        "variance": variance,
        "mean_entropy": compute_mean_entropy(entropy_sequence, mask=mask),
        "seq_length": len(entropy_sequence),
    }

    return edis, metrics


def compute_edis_batch(
    entropy_sequences: List[ArrayLike],
    masks: Optional[List[ArrayLike]] = None,
    **kwargs,
) -> List[float]:
    out = []
    for i, seq in enumerate(entropy_sequences):
        mask = masks[i] if masks is not None else None
        out.append(
            compute_edis_from_sequence(seq, mask=mask, return_details=False, **kwargs)
        )
    return out
