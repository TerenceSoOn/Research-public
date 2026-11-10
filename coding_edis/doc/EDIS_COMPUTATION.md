# EDIS (Entropy Dynamics Instability Score) 计算方法文档

> **版本**: 1.0  
> **最后更新**: 2026-02-02  
> **相关文件**: `edis_core.py`, `verl/utils/entropy_dynamics.py`

---

## 目录

1. [概述](#1-概述)
2. [核心公式](#2-核心公式)
3. [Spike 检测方法](#3-spike-检测方法)
4. [完整计算流程](#4-完整计算流程)
5. [代码实现](#5-代码实现)
6. [参数配置](#6-参数配置)
7. [使用示例](#7-使用示例)
8. [实验结果](#8-实验结果)

---

## 1. 概述

### 1.1 什么是 EDIS？

**EDIS (Entropy Dynamics Instability Score)** 是一个基于熵动态变化的响应质量评估指标。

核心发现：
- **错误响应的熵突变次数是正确响应的 1.8-2.0 倍**
- 正确推理：从"不确定"快速收敛到"确定"，熵变化**平滑**
- 错误推理：反复"卡顿"在不确定状态，熵突变**频繁**

### 1.2 与现有方法的区别

| 现有方法 | 关注点 | EDIS | 关注点 |
|---------|--------|------|--------|
| 熵最小化 (RENT等) | 熵有多低 | EDIS | 熵怎么变化 |
| 熵正则化 (AER等) | 防止熵崩塌 | EDIS | 突变次数/方差 |
| Token选择 (Qwen) | 哪些token熵高 | EDIS | 整体变化模式 |

### 1.3 设计原理

```
低 EDIS → 推理稳定、流畅 → 更可能正确
高 EDIS → 推理不稳定、多次犹豫 → 更可能错误
```

---

## 2. 核心公式

### 2.1 EDIS 公式

```
EDIS = spike × (1 + variance)
```

其中：
- `spike`: 熵突变次数（混合方法计算）
- `variance`: 熵序列的方差

### 2.2 Spike 计算公式

```
spike = 0.5 × (burst_spike + peak_trough_spike)
```

其中：
- `burst_spike`: 突发 spike（检测持续恶化）
- `peak_trough_spike`: 峰谷 spike（检测 V 型反弹）

### 2.3 公式设计原理

```
┌─────────────────────────────────────────────────────────────────┐
│  EDIS = spike × (1 + variance)                                   │
│                                                                  │
│  设计原理:                                                        │
│    - spike 多 → 模型经常突然犹豫 → 不好                           │
│    - variance 大 → 熵值波动剧烈 → 不好                            │
│    - 乘法组合 → 两者相互放大不稳定性                              │
│                                                                  │
│  筛选时: 保留 EDIS 最低的样本（最稳定）                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Spike 检测方法

### 3.1 方法对比

| 检测方法 | 医学比喻 | 捕捉的问题 |
|---------|---------|-----------|
| 固定阈值 | 血压一次超过180就报警 | 单步突变 |
| **突发 spike** | 连续5天血压净增长超过阈值就报警 | 持续恶化趋势 |
| **峰谷 spike** | 血压从最低点反弹超过30就报警 | V型反弹 |
| **突发+峰谷** | 综合考虑趋势和反弹 | ✅ 最全面 |

### 3.2 突发 Spike (Burst Spike) - 检测"持续恶化"

**原理**: 在滑动窗口(5个token)内，如果熵的累计增长超过阈值，说明模型在"持续犹豫"

**公式**:
```
burst_spike = Σ I(entropy[i + window] - entropy[i] > burst_threshold)
```

**示例**:
```
序列: [1.0, 1.2, 1.4, 1.6, 1.8, 2.6]
位置0: seq[5] - seq[0] = 2.6 - 1.0 = 1.6 > 1.36 → spike!

✓ [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5]  # 缓慢但持续上升
✗ [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]              # 反复跳跃但无趋势
```

**代码实现**:

```python
def compute_burst_spike(entropy_sequence, threshold=1.36, window=5, mask=None):
    """
    计算突发 Spike：检测窗口内累计熵增长超过阈值的情况。
    
    Args:
        entropy_sequence: 熵序列，每个位置的熵值
        threshold: 累计增长阈值，默认 1.36
        window: 滑动窗口大小，默认 5
        mask: 可选的掩码，1=有效，0=填充
    
    Returns:
        突发 spike 次数
    """
    # 转换为 numpy 数组
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    # 应用掩码
    if mask is not None:
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
```

### 3.3 峰谷 Spike (Peak-to-Trough Spike) - 检测"V型反弹"

**原理**: 检测熵从历史最低点反弹超过阈值的情况，捕捉"以为懂了但又糊涂了"的模式

**公式**:
```
peak_trough_spike = Σ I(entropy[i] - min(entropy[:i]) > peak_threshold)
```

**示例**:
```
序列: [2.0, 1.5, 1.0, 0.5, 0.8, 1.5, 2.0]
位置5: min so far = 0.5, current = 1.5, rebound = 1.0 < 1.3 → 无spike
位置6: min so far = 0.5, current = 2.0, rebound = 1.5 > 1.3 → spike!

✓ [2.0, 1.0, 0.5, 1.0, 2.0]  # V型反弹：从低点大幅上升
✓ [0.5, 0.3, 0.8, 0.2, 1.6]  # 多次反弹：反复"以为好了又差了"
✗ [0.5, 0.6, 0.7, 0.8, 0.9]  # 单调上升但无明显低点反弹
```

**代码实现**:

```python
def compute_peak_spike(entropy_sequence, threshold=1.33, mask=None):
    """
    计算峰谷 Spike：检测从历史最低点反弹超过阈值的情况。
    
    Args:
        entropy_sequence: 熵序列，每个位置的熵值
        threshold: 反弹阈值，默认 1.33
        mask: 可选的掩码，1=有效，0=填充
    
    Returns:
        峰谷 spike 次数
    """
    # 转换为 numpy 数组
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    # 应用掩码
    if mask is not None:
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
```

### 3.4 混合 Spike 计算

**公式**:
```
spike = 0.5 × (burst_spike + peak_trough_spike)
```

**代码实现**:

```python
def compute_hybrid_spike(entropy_sequence, burst_threshold=1.36, 
                         peak_threshold=1.33, window=5, mask=None):
    """
    计算混合 Spike：突发 + 峰谷的加权平均。
    
    返回: (combined_spike, burst_spike, peak_spike)
    """
    burst_spike = compute_burst_spike(
        entropy_sequence, threshold=burst_threshold, window=window, mask=mask
    )
    
    peak_spike = compute_peak_spike(
        entropy_sequence, threshold=peak_threshold, mask=mask
    )
    
    # 加权平均
    combined_spike = 0.5 * (burst_spike + peak_spike)
    
    return combined_spike, burst_spike, peak_spike
```

---

## 4. 完整计算流程

### 4.1 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    EDIS 完整计算流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入: entropy_sequence (token级熵序列)                         │
│                                                                 │
│         ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Step 1: 计算 Burst Spike                 │                   │
│   │   burst = Σ I(seq[i+5] - seq[i] > 1.36) │                   │
│   │   检测"持续恶化"模式                      │                   │
│   └─────────────────────────────────────────┘                   │
│         ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Step 2: 计算 Peak Spike                  │                   │
│   │   peak = Σ I(seq[i] - min < seq[i]) > 1.33) │               │
│   │   检测"V型反弹"模式                       │                   │
│   └─────────────────────────────────────────┘                   │
│         ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Step 3: 计算混合 Spike                   │                   │
│   │   spike = 0.5 × (burst + peak)          │                   │
│   └─────────────────────────────────────────┘                   │
│         ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Step 4: 计算熵方差                       │                   │
│   │   variance = Var(entropy_sequence)       │                   │
│   └─────────────────────────────────────────┘                   │
│         ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Step 5: 计算 EDIS                        │                   │
│   │   EDIS = spike × (1 + variance)         │                   │
│   └─────────────────────────────────────────┘                   │
│         ↓                                                       │
│   输出: EDIS 分数 (越低越好)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 完整代码实现

```python
def compute_edis_from_sequence(entropy_sequence, burst_threshold=1.36,
                               peak_threshold=1.33, window=5, mask=None):
    """
    从熵序列直接计算 EDIS（一站式接口）。
    
    Args:
        entropy_sequence: 熵序列
        burst_threshold: 突发检测阈值，默认 1.36
        peak_threshold: 峰谷检测阈值，默认 1.33
        window: 滑动窗口大小，默认 5
        mask: 可选的掩码
    
    Returns:
        (edis, metrics_dict) 元组
    """
    import numpy as np
    
    # 转换为 numpy 数组
    if isinstance(entropy_sequence, list):
        entropy_sequence = np.array(entropy_sequence)
    
    # Step 1-3: 计算混合 Spike
    # Step 1: Burst Spike
    burst_spike = 0
    if len(entropy_sequence) >= window + 1:
        for i in range(len(entropy_sequence) - window):
            if entropy_sequence[i + window] - entropy_sequence[i] > burst_threshold:
                burst_spike += 1
    
    # Step 2: Peak Spike
    peak_spike = 0
    if len(entropy_sequence) >= 2:
        historical_min = entropy_sequence[0]
        for i in range(1, len(entropy_sequence)):
            if entropy_sequence[i] - historical_min > peak_threshold:
                peak_spike += 1
            if entropy_sequence[i] < historical_min:
                historical_min = entropy_sequence[i]
    
    # Step 3: 混合 Spike
    combined_spike = 0.5 * (burst_spike + peak_spike)
    
    # Step 4: 计算熵方差
    variance = float(np.var(entropy_sequence)) if len(entropy_sequence) >= 2 else 0.0
    
    # Step 5: 计算 EDIS
    edis = combined_spike * (1 + variance)
    
    metrics = {
        "edis": edis,
        "spike": combined_spike,
        "burst_spike": burst_spike,
        "peak_spike": peak_spike,
        "variance": variance,
        "mean_entropy": float(np.mean(entropy_sequence)),
        "seq_length": len(entropy_sequence),
    }
    
    return edis, metrics
```

---

## 5. 代码实现

### 5.1 文件结构

```
verl-entropy/
├── edis_core.py                    # 独立的 EDIS 计算模块（轻量级）
├── verl/utils/entropy_dynamics.py  # 完整实现（集成到 verl 框架）
└── EDIS_COMPUTATION.md             # 本文档
```

### 5.2 使用 `edis_core.py`（推荐）

```python
from edis_core import compute_edis_from_sequence

# 简单用法
entropy_seq = [0.5, 0.8, 1.2, 1.8, 1.5, 2.1, 0.9]
edis, metrics = compute_edis_from_sequence(entropy_seq)

print(f"EDIS: {edis:.4f}")
print(f"Spike: {metrics['spike']}")
print(f"Variance: {metrics['variance']:.4f}")
```

### 5.3 使用 `entropy_dynamics.py`（verl 框架）

```python
from verl.utils.entropy_dynamics import (
    compute_edis_from_entropys,
    compute_entropy_metrics,
    apply_edis_weighting_to_advantages,
)

# 批量计算 (PyTorch Tensor)
import torch

entropys = torch.randn(8, 512)  # [batch_size, seq_len]
response_masks = torch.ones(8, 512)

# 计算 EDIS
edis_scores = compute_edis_from_entropys(entropys, response_masks)

# 计算完整指标
metrics = compute_entropy_metrics(entropys, response_masks)
```

---

## 6. 参数配置

### 6.1 默认参数（经过 1700+ 种组合测试优化）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `burst_threshold` | 1.36 | 突发检测阈值：窗口内累计增长超过此值触发 |
| `peak_threshold` | 1.33 | 峰谷检测阈值：从历史最低反弹超过此值触发 |
| `window` | 5 | 突发检测的滑动窗口大小 |

### 6.2 参数搜索结果

| 参数 | 最优值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| burst_threshold | 1.36 | 0.7-2.0 | 窗口首尾之差阈值 |
| burst_window | 5 | 3-8 | 滑动窗口大小 |
| peak_threshold | 1.33 | 0.8-1.8 | 从历史最低反弹阈值 |
| weight | 0.5 | - | 乘以0.5使结果可解释为"熵激变平均次数" |

### 6.3 在 verl 中的配置

```bash
# run_grpo.sh 中的配置

# Spike 计算参数
entropy_burst_threshold=1.36
entropy_peak_threshold=1.33
entropy_window=5
```

---

## 7. 使用示例

### 7.1 基本示例

```python
from edis_core import compute_edis_from_sequence

# 示例 1: 稳定的熵序列（低 EDIS）
stable_seq = [0.5, 0.6, 0.55, 0.58, 0.52, 0.54]
edis_stable, _ = compute_edis_from_sequence(stable_seq)
print(f"稳定序列 EDIS: {edis_stable:.4f}")  # 输出较低的 EDIS

# 示例 2: 不稳定的熵序列（高 EDIS）
unstable_seq = [0.5, 1.2, 0.8, 2.1, 0.6, 2.5]
edis_unstable, _ = compute_edis_from_sequence(unstable_seq)
print(f"不稳定序列 EDIS: {edis_unstable:.4f}")  # 输出较高的 EDIS
```

### 7.2 批量处理示例

```python
from edis_core import compute_edis_batch

sequences = [
    [0.5, 0.6, 0.55, 0.58],  # 稳定
    [0.5, 1.2, 0.8, 2.1],   # 不稳定
    [1.0, 1.5, 2.0, 2.5],   # 持续上升
]

edis_scores, all_metrics = compute_edis_batch(sequences)

for i, (edis, metrics) in enumerate(zip(edis_scores, all_metrics)):
    print(f"序列 {i}: EDIS={edis:.4f}, spike={metrics['spike']}")
```

### 7.3 用于筛选响应

```python
from edis_core import compute_edis_from_sequence

def select_best_response(responses_with_entropy):
    """
    从多个响应中选择 EDIS 最低的（最可能正确的）。
    
    Args:
        responses_with_entropy: [(response_text, entropy_sequence), ...]
    
    Returns:
        最佳响应
    """
    best_response = None
    best_edis = float('inf')
    
    for response_text, entropy_seq in responses_with_entropy:
        edis, _ = compute_edis_from_sequence(entropy_seq)
        if edis < best_edis:
            best_edis = edis
            best_response = response_text
    
    return best_response, best_edis
```

---

## 8. 实验结果

### 8.1 区分度对比

| 方法 | 准确率 | 区分度 | 相对提升 |
|------|--------|--------|----------|
| 原始（无筛选） | 32.3% | 1.00x | - |
| 固定阈值 1.0 | 51.09% | 1.62x | baseline |
| 只用 burst (1.36) | 51.88% | 1.83x | +0.8% |
| 只用 peak-trough (1.33) | 51.72% | 1.78x | +0.6% |
| **突发+峰谷 (EDIS)** | **52.19%** | **1.84x** | **+1.1%** |

### 8.2 跨模型验证

| 模型 | 正确样本 spike | 错误样本 spike | 比值 | p-value | Cohen's d |
|------|---------------|---------------|------|---------|-----------|
| Qwen2.5-Math-1.5B | 56.4±23.5 | 92.6±37.3 | **1.64x** | 1.6e-22 | 1.03 |
| Qwen3-1.7B-Base | 71.0±36.5 | 134.9±45.6 | **1.90x** | 1.2e-32 | 1.22 |

**结论**: 效应量 Cohen's d > 1.0，非常显著

### 8.3 跨训练步骤一致性

```
Qwen2.5 各步骤的 错误/正确 突变比值:
Step 0:   1.40x
Step 100: 1.70x
Step 200: 1.44x
Step 300: 1.64x
Step 400: 1.66x
Step 500: 1.56x
Step 600: 1.60x
```

**结论**: 训练全程稳定在 1.4-1.7x

---

## 附录 A: 数学符号说明

| 符号 | 含义 |
|------|------|
| $E$ | 熵序列 entropy_sequence |
| $E_i$ | 位置 i 的熵值 |
| $B$ | burst_spike 计数 |
| $P$ | peak_spike 计数 |
| $S$ | 混合 spike = 0.5 × (B + P) |
| $\sigma^2$ | 熵方差 variance |
| $\text{EDIS}$ | S × (1 + $\sigma^2$) |

## 附录 B: 引用

如果使用本方法，请引用：

```bibtex
@article{edis2026,
  title={EDIS: Entropy Dynamics Instability Score for Response Quality Assessment},
  year={2026}
}
```

---

*文档结束*
