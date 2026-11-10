# old design
1. 采样32
2. 聚类(borda voting)
3. 筛选8个 - 正负样本比例(随机选)
4. 重新聚类算奖励/**直接沿用奖励(0,1)**/加权优势
5. grpo

- borda voting影响的重新跑
- 样本比例
- 加权优势
- 重新聚类算奖励(EMPO)


# old
A: EMPO: Generate 8 samples -> EMPO clustering and calculating EMPO rewards -> use the reward directly to calculate adv -> train
B: EMPO + Oversampled: Generate 32 samples -> EMPO clustering and calculating binary rewards -> select 8 samples randomly -> use the reward directly to calculate adv -> train
C: EMPO+Oversampled + sample selection: Generate 32 samples -> EMPO clustering and calculating 1/0 rewards -> select 8 samples randomly, 4 from the largest cluster, 4 from the rest clusters, both picked randomly -> use the 1/0 reward directly to calculate adv -> train
D: EMPO+Oversampled + sample selection: Generate 32 samples -> EMPO clustering and calculating 1/0 rewards -> select 8 samples randomly, 4 from the largest cluster, 4 from the rest clusters, both picked randomly -> use the 1/0 reward directly to calculate adv -> use entrpy to add weight to the adv(for all responses in a batch, record entropy for each, mark the top-20% value of the entropy, then for the entorpies of selected 8 samples, for samples from largest cluster, those with low entropy gain high weights; for samples from the rest clusters, those with high entropy gain high weights) -> train

# new

## logic:
- Parameters are reduced to: 'oversampled' and 'weighted'
- Sample selection of oversampled is removed, always select samples randomly(8/32)
- weighting machenism remains the same: 'For all responses in a batch, record entropy for each, mark the top-20% value of the entropy, then for the entorpies of selected 8 samples, for samples from largest cluster, those with low entropy gain high weights; for samples from the rest clusters, those with high entropy gain high weights'

## settings:
1. oversampled = off,  weighted = off
2. oversampled = on,  weighted = off
3. oversampled = off,  weighted = on
4. oversampled = on,  weighted = on

