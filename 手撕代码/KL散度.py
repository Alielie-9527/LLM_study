# 个人笔记参考：https://alielie-9527.github.io/KL散度/
import torch
import torch.nn.functional as F

def compute_kl_divergence(logp_policy: torch.Tensor, 
                          logp_ref: torch.Tensor, 
                          estimator: str = "kl3") -> torch.Tensor:
    """
    计算近似 KL 散度
    
    参数:
        logp_policy: 当前正在训练的模型生成的 Token 的对数概率 (Shape: [Batch_size, Seq_Len])
        logp_ref: 参考模型生成的相同 Token 的对数概率 (Shape: [Batch_size, Seq_Len])
        estimator: 估计器类型 ("kl1", "kl2", "kl3")
        
    返回:
        kl_penalty: KL 惩罚值 (Shape: [Batch_size, Seq_Len])
    """
    
    # 计算对数概率之差 x
    # x = log(P) - log(Q)
    log_ratio = logp_policy - logp_ref
    
    if estimator == "kl1":
        # KL1: 最基础的近似 (均值正确，但单个值可能为负，方差大)
        kl_penalty = log_ratio
        
    elif estimator == "kl2":
        # KL2: 泰勒展开近似 (永远非负，方差较小，但有偏差)
        kl_penalty = 0.5 * (log_ratio ** 2)
        
    elif estimator == "kl3":
        # KL3: Schulman (永远非负，方差最小，现代大模型标配)
        # 公式: e^(-x) + x - 1
        kl_penalty = torch.exp(-log_ratio) + log_ratio - 1.0
        
    else:
        raise ValueError("未知的 KL 估计器类型！")
        
    return kl_penalty

# ================= 测试代码 =================
# 假设我们有一个 Batch，里面包含 3 个 Token 的对数概率
# 对数概率是负数，因为概率在 0~1 之间，log(0.x) < 0
logp_policy = torch.tensor([-0.1, -0.5, -2.0])  # 当前模型比较自信、一般、很不自信
logp_ref    = torch.tensor([-0.2, -0.5, -1.0])  # 参考模型的信心

print(f"Log_Ratio (x): {logp_policy - logp_ref}")

kl1_val = compute_kl_divergence(logp_policy, logp_ref, estimator="kl1")
kl2_val = compute_kl_divergence(logp_policy, logp_ref, estimator="kl2")
kl3_val = compute_kl_divergence(logp_policy, logp_ref, estimator="kl3")

print("-" * 30)
print(f"KL1 (原始) 估计: {kl1_val}")
print(f"KL2 (平方) 估计: {kl2_val}")
print(f"KL3 (指数) 估计: {kl3_val}")