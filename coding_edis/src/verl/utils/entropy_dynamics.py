from typing import Optional
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_spike_count(
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 0.3,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> int:
    entropy_sequence = _to_numpy(entropy_sequence)
    mask = _to_numpy(mask) if mask is not None else None

    if len(entropy_sequence) < 2:
        return 0

    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0
        entropy_sequence = entropy_sequence[valid_indices]

    diffs = np.diff(entropy_sequence)
    return int(np.sum(diffs > threshold))


def compute_burst_spike(
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 1.36,
    window: int = 5,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> int:
    entropy_sequence = _to_numpy(entropy_sequence)
    mask = _to_numpy(mask) if mask is not None else None

    if mask is not None:
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
    entropy_sequence: np.ndarray | torch.Tensor,
    threshold: float = 1.33,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> int:
    entropy_sequence = _to_numpy(entropy_sequence)
    mask = _to_numpy(mask) if mask is not None else None

    if mask is not None:
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
    entropy_sequence: np.ndarray | torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> tuple[float, int, int]:
    burst_spike = compute_burst_spike(
        entropy_sequence, threshold=burst_threshold, window=window, mask=mask
    )

    peak_spike = compute_peak_spike(
        entropy_sequence, threshold=peak_threshold, mask=mask
    )

    combined_spike = 0.5 * (burst_spike + peak_spike)

    return combined_spike, burst_spike, peak_spike


def compute_entropy_variance(
    entropy_sequence: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> float:
    entropy_sequence = _to_numpy(entropy_sequence)
    mask = _to_numpy(mask) if mask is not None else None

    if len(entropy_sequence) < 2:
        return 0.0

    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 2:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]

    return float(np.var(entropy_sequence))


def compute_mean_entropy(
    entropy_sequence: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> float:
    entropy_sequence = _to_numpy(entropy_sequence)
    mask = _to_numpy(mask) if mask is not None else None

    if len(entropy_sequence) < 1:
        return 0.0

    if mask is not None:
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 1:
            return 0.0
        entropy_sequence = entropy_sequence[valid_indices]

    return float(np.mean(entropy_sequence))


def compute_edis(
    spike_count: float,
    entropy_variance: float,
    mean_entropy: float = None,
    seq_length: int = None,
    **kwargs,
) -> float:
    edis = spike_count * (1 + entropy_variance)
    return float(edis)


def compute_entropy_metrics(
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
    **kwargs,
) -> dict:
    batch_size = entropys.shape[0]

    mean_entropys = []
    spike_counts = []
    entropy_variances = []
    burst_spikes = []
    peak_spikes = []

    for i in range(batch_size):
        entropy_seq = entropys[i]
        mask = response_masks[i] if response_masks is not None else None

        mean_entropy = compute_mean_entropy(entropy_seq, mask=mask)
        entropy_var = compute_entropy_variance(entropy_seq, mask=mask)

        combined_spike, burst_spike, peak_spike = compute_hybrid_spike(
            entropy_seq,
            burst_threshold=burst_threshold,
            peak_threshold=peak_threshold,
            window=window,
            mask=mask,
        )

        mean_entropys.append(mean_entropy)
        spike_counts.append(combined_spike)
        entropy_variances.append(entropy_var)
        burst_spikes.append(burst_spike)
        peak_spikes.append(peak_spike)

    edis_scores = [
        compute_edis(spike_counts[i], entropy_variances[i])
        for i in range(batch_size)
    ]

    return {
        "mean_entropys": mean_entropys,
        "spike_counts": spike_counts,
        "entropy_variances": entropy_variances,
        "edis_scores": edis_scores,
        "burst_spikes": burst_spikes,
        "peak_spikes": peak_spikes,
    }


def compute_edis_from_entropys(
    entropys: torch.Tensor,
    response_masks: torch.Tensor,
    burst_threshold: float = 1.36,
    peak_threshold: float = 1.33,
    window: int = 5,
) -> torch.Tensor:
    metrics = compute_entropy_metrics(
        entropys,
        response_masks,
        burst_threshold=burst_threshold,
        peak_threshold=peak_threshold,
        window=window,
    )
    return torch.tensor(metrics["edis_scores"], device=entropys.device)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def edis_weighting(
    edis_scores: np.ndarray,
    correct_mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    z = zscore(edis_scores)
    s = np.where(correct_mask > 0, -z, z)
    s = s / max(temperature, 1e-6)
    exps = np.exp(s - s.max())
    return exps / exps.sum()


def apply_edis_weighting_to_advantages(
    advantages: np.ndarray,
    edis_scores: np.ndarray,
    correct_mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    weights = edis_weighting(edis_scores, correct_mask, temperature=temperature)
    return advantages * weights
