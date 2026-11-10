
election_family:
- edis_topk = classic EDIS filtering(直接选8个最稳定的)
- balanced_largest_vs_rest = tutor-style balanced selection(按比例分)
- random_topk = oversampled EMPO behavior(随机)


balanced_selection_method: only used when selection_family=balanced_largest_vs_rest
- A: 正负样本对半选择, 正确的选择最稳定的, 错误的也选最稳定的 
- B: 正负对半, 正确的最稳定的, 错误的选最不稳定的  
- C: 正负对半, 正确最稳定, 错误随机选    
- D: 正负对半, 正确随机选, 错误随机选


reward_method:
- recompute_empo = EMPO再算一次
- direct_vote_tag = 直接沿用voting的结果
- edis_weighted_vote_tag = edis加权后的voting结果


vote_weighting:
- off = normal majority vote
- edis_borda = weighted vote when deciding winner/tag

majority_share: how much of K comes from the largest cluster, 0.5 means half from largest cluster, half from rest  


underfilled_policy: currently only skip_prompt  

vote_weight_epsilon:  epsilon for inverse-EDIS weighting in edis_borda  