# PPO的损失函数
import torch
import torch.nn as nn

def PPO_Loss(old_log_probs,new_log_probs,values,targets,advantages,clip_esplion=0.2,critic_coef=0.5,entropy_coef=0.01):
        """
        Args:
        # 出于计算稳定性来取log
            old_log_probs: 旧策略的 log π(a|s), shape: [batch]
            new_log_probs: 新策略的 log π(a|s), shape: [batch]
            values:        Critic 预测的价值 V(s), shape: [batch]
            targets:       价值目标 (reward + discount * V_next),shape: [batch]
            advantages:    优势函数 A(s,a), shape: [batch]
            clip_epsilon:    # PPO clipping 参数
            critic_coef:      # Critic 损失权重
            entropy_coef:    # 熵正则权重
        """
        ## 计算 Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 进行裁剪（PPO 的核心）
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_esplion, 1.0 + clip_esplion) * advantages

        # Actor 损失（取最小的剪枝项并对批次求平均）
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic 损失（均方误差）
        critic_loss = nn.MSELoss()(values, targets)

        # 熵（entropy）计算：H(p) = -sum_a p(a) log p(a)
        # 说明：
        # - 若 `new_log_probs` 含有动作维（即对每个动作都有 log-prob），对最后一维求和以得到每个样本的熵。
        # - 若 `new_log_probs` 只是所采样动作的 log-prob（标量），则无法从单个采样估计完整分布熵，下面的实现会退化为对该标量项的近似。
        # 熵的作用：在优化中我们希望策略保持一定的不确定性以促进探索，避免过早收敛到次优策略。
        # 因此在最终 loss 中通常是减去熵项（因为我们最小化 loss，但希望最大化熵）。
        entropy_term = - (new_log_probs.exp() * new_log_probs)
        if entropy_term.dim() > 1:
            entropy = entropy_term.sum(dim=-1).mean()
        else:
            entropy = entropy_term.mean()

        # 总损失：最小化 actor + critic，并且最大化熵（通过在 loss 中减去熵项实现）
        total_loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

        # 返回总损失和各个分项，方便监控
        return total_loss, {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": entropy}