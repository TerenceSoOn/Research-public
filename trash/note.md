#
EMPO奖励在选完之后应该重新算

#
算法流程:
1. Weighted Majority Voting (Borda voting)得到voting标签
2. sample selection: 根据voting标签选择样本
    - A: 正负样本对半选择, 正确的选择最稳定的, 错误的也选最稳定的
    - B: 正负对半, 正确的最稳定的, 错误的选最不稳定的 !
    - C: 正负对半, 正确最稳定, 错误随机选 !
    - D: 正负对半, 正确随机选, 错误随机选
3. 计算奖励:
    - A: EMPO再算一次
    - B: 直接沿用voting的结果 !
    - C: edis加权后的voting结果 !
4. 计算优势

后面加!的是一定要试的

#
要关注论文里的这一部分
For confidence-based methods (Entropy, SC, EDIS), we use Score-Weighted Borda aggregation: each response casts a weighted vote for its predicted answer, with the highest-weighted answer selected. For metrics where lower values indicate higher confidence (Entropy, EDIS), we use inverse weighting wi = (si + ε)−1 with ε = 0.1; for SC, we use wi = si directly. Experiments use Qwen2.5-Math- 1.5B across four benchmarks, generating m ∈ {4, 8, 16} candidates per problem, with results averaged across three temperatures.


#
EDIS weighting: weighting for calculating clusters(eg: 1.0 sample with high edis stability become 2.0 time of itself when forming clusters and calculating the majority), instead of rewards.