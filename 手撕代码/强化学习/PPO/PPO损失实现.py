import torch
import torch.nn as nn

def PPO_Loss(old_log_probs,new_log_probs,values,targets,advantages,clip_esplion=0.2,critic_coef=0.5,entropy_coef=0.01):
    '''
    计算PPO损失函数

    Args:
        old_log_probs: 旧策略的 log π(a|s), shape: [batch]
        new_log_probs: 新策略的 log π(a|s), shape: [batch]
        values:        Critic 预测的价值 V(s), shape: [batch]
        targets:       价值目标 (reward + discount * V_next),shape: [batch]
        advantages:    优势函数 A(s,a), shape: [batch]
        clip_epsilon:    PPO clipping 参数
        critic_coef:      Critic 损失权重
        entropy_coef:    熵正则权重

    Returns:
        total_loss: 总损失
        loss_info: 各个损失项的详细信息
    '''
    ## 计算Ratio 重要性采样
    ratio = torch.exp(new_log_probs - old_log_probs)

    ## 进行PPO裁剪
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio,1 - clip_esplion,1 + clip_esplion) *advantages

    # actor loss 
    actor_loss = -torch.min(surr1,surr2).mean()

    # critic loss 
    critic_loss = nn.MSELoss()(values,targets)

    # 熵（entropy）计算：H(p) = -sum_a p(a) log p(a)
    # 说明：
    # - 若 `new_log_probs` 含有动作维（即对每个动作都有 log-prob），对最后一维求和以得到每个样本的熵。
    # - 若 `new_log_probs` 只是所采样动作的 log-prob（标量），则无法从单个采样估计完整分布熵，下面的实现会退化为对该标量项的近似。
    # 熵的作用：在优化中我们希望策略保持一定的不确定性以促进探索，避免过早收敛到次优策略。
    # 因此在最终 loss 中通常是减去熵项（因为我们最小化 loss，但希望最大化熵）。
    entropy_term = -(new_log_probs * torch.log(new_log_probs)).mean()
    total_loss = actor_loss  + critic_coef * critic_loss - entropy_coef * entropy_term

    return total_loss,{"actor_loss":actor_loss,"critic_loss":critic_loss,"entropy":entropy_term}