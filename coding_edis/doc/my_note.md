# EDIS
## EDIS指标计算
**Token级Entropy**: $H_{t} = -\sum_{v \in \mathcal{V}} \pi_{\theta}(v | x, y_{<t}) \log \pi_{\theta}(v | x, y_{<t})$

**Burst Spikes**: 
- $S_{burst} = \sum_{t=1}^{T-w} \mathbb{I}(H_{t+w} - H_t > \tau_b)$
- 预设观测窗口w, 将在w步之后的token的熵与当前熵进行对比, 如大于一定量测记为一次Burst Spikes


**Rebound Spikes**:
- $S_{rebound} = \sum_{t=2}^{T} \mathbb{I}(H_t - \min_{s<t} H_s > \tau_r)$
- 维护历史最低熵, 如果某一token的熵高于最低熵一定量, 记录一次Rebound Spike


**Entropy Variance**: $\text{Var}(H) = \frac{1}{T} \sum_{t=1}^{T} (H_t - \bar{H})^2$


**EDIS Score Calculation**: 
- $EDIS(H) = S(H) \cdot (1 + \text{Var}(H)) \quad$
- Where $S(H) = \frac{1}{2}(S_{burst} + S_{rebound})$



## 流程
EDIS Weighting:
1. 采样, 记录Burst Spikes与Rebound Spikes
2. 计算EDIS Score
3. 对EDIS Score进行标准化($s_i = \begin{cases} -z_i & \text{if correct} \\ z_i & \text{if incorrect} \end{cases}$)
4. 根据$s_i$计算权重
    - 对于正确答案, $s_i$越小代表越自信, 越应该赋予高权重, 筛除蒙对的答案
   - 对于错误答案, $s_i$越大代表越混乱, 赋予高权重来加强对错误答案的修正
5. 对权重进行softmax归一化得到最终权重 ($w_i^{norm} = \frac{w_i}{\sum_{j \in \mathcal{G}_i} w_j} \cdot |\mathcal{G}_i|$)
----
EDIS Filtering
1. 进行超采样
2. 取最稳定的正确答案集和最混乱的错误答案集中取样
3. 直到batch被填满, 丢弃中间样本
---
与GRPO结合的weighting流程:
1. 采样, 记录Burst Spikes与Rebound Spikes
2. 计算双重信号:
    - 基础奖励
    - GRPO Raw Advantage
    - EDIS Score
3. 信号融合:
    - 对EDIS Score进行标准化
    - 根据$s_i$计算权重
    - 对权重进行softmax归一化
    - $\tilde{A}_{i} = A_{i} \cdot w_{i}^{norm}$, 应用与Standard GRPO Objective

## 实验

### 1. Best Of N
- 在样本数量上升时(m: 1~16), EDIS的准确性也在上升(Figure4)
- 在同等样本数下(m=4,8,16), EDIS要优于其他选择方法(Mean, Majority-Voting, Sequence-Entropy, Self-certainty)(Table1)

### 2. EDIS For RL
#### 配置
- Qwen2.5-Math-1.5B
- NuminaMath-20K
- 500 steps
- T = 0.6/0.2

#### 实验结果(T=0.6)
EDIS Weighting + Filtering 与 GRPO做对比:
- maj@8 60.8% + 5.4%
- mean@8 53.8% +8.1%

#### 实验结果(T=0.2)
maj@8 / mean@8
- GRPO: 59.1% / 55.0%
- Over Sampling: 62.1% (+3.0) / 58.4% (+3.4)
- EDIS Flitering: 62.6% (+3.5) / 60.6% (+5.6)
- EDIS Weighting: 66.8% (+7.7) / 62.2% (+7.2)
- Filtering + Weighting: 66.5% (+7.4) / 62.8% (+7.8)

## 局限性
- 仅用于数学问题
- 通用性需进一步讨论
- RL验证需进一步讨论