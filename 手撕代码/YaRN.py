import torch
import math

def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """
    计算 YaRN 的温度缩放系数 mscale。
    公式：mscale(k) = 0.1 * mscale * ln(k) + 1.0
    当 scale <= 1 时，无需扩展，直接返回 1.0。
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_freqs_cis_yarn(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_seq_len: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
):
    """
    使用 YaRN 方法预计算 RoPE 的频率张量。

    Args:
        dim:                  每个 head 的维度 (head_dim)
        end:                  推理时支持的最大序列长度（扩展后）
        theta:                频率基底，默认 10000.0
        scaling_factor:       上下文长度扩展倍数 k
        original_max_seq_len: 模型训练时的原始最大序列长度 L
        beta_fast:            高频/中频分界阈值（转数 r_i > beta_fast 为高频区，直接外推）
        beta_slow:            低频/中频分界阈值（转数 r_i < beta_slow 为低频区，线性插值）
        mscale:               温度缩放基础超参数
        mscale_all_dim:       对所有维度的额外 mscale 缩放（用于部分变体）
    """
    # -------------------------------------------------------
    # Step 1：计算原始基础逆频率
    # inv_freq[i] = theta^(-2i/d),  shape: [dim/2]
    # -------------------------------------------------------
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # -------------------------------------------------------
    # Step 2：NTK-by-parts 分频段修正 inv_freq
    # 根据每个维度在训练窗口 L 内的转数 r_i = L * inv_freq[i] / (2π)
    # 决定高频/中频/低频的处理策略
    # -------------------------------------------------------
    freq_extra = inv_freq.clone()           # 高频策略：保持不变（外推）
    freq_inter = inv_freq / scaling_factor  # 低频策略：缩小 k 倍（线性插值）

    # 计算转数 r_i = L / λ_i = L * inv_freq[i] / (2π)
    r_i = original_max_seq_len * inv_freq / (2 * math.pi)

    # 平滑过渡系数 γ_i ∈ [0, 1]
    #   r_i < beta_slow → 低频区，γ → 0，采用插值
    #   r_i > beta_fast → 高频区，γ → 1，采用外推
    #   beta_slow ≤ r_i ≤ beta_fast → 中频区，线性混合
    gamma = (r_i - beta_slow) / (beta_fast - beta_slow)
    gamma = torch.clamp(gamma, 0.0, 1.0)

    # 混合两种策略：θ_i' = (1 - γ_i) * θ_i/k + γ_i * θ_i
    inv_freq_yarn = (1.0 - gamma) * freq_inter + gamma * freq_extra

    # -------------------------------------------------------
    # Step 3：计算 mscale 温度增益系数
    # 用于后续对 Q、K 进行缩放，抵消序列变长导致的注意力分散
    # -------------------------------------------------------
    _mscale = float(
        yarn_get_mscale(scaling_factor, mscale)
        / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )

    # -------------------------------------------------------
    # Step 4：生成位置序列，计算最终复数频率张量
    # -------------------------------------------------------
    t = torch.arange(end, device=inv_freq_yarn.device, dtype=torch.float32)

    # 外积：每个位置 × 每个维度的频率 → [end, dim/2]
    freqs = torch.outer(t, inv_freq_yarn)

    # 转换为极坐标复数张量: [end, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis, _mscale


def apply_rotary_emb_yarn(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    mscale: float = 1.0,
):
    """
    将 YaRN RoPE 应用于 Query 和 Key，并施加 mscale 温度增益。

    mscale 直接乘在 xq/xk 上，等效于将 Attention logits 放大 mscale² 倍，
    从而使 Softmax 输出更尖锐，抵消长序列带来的注意力分散问题：
        Q' = mscale * Q,  K' = mscale * K
        Q'K'^T = mscale² * QK^T
    """
    # reshape 为复数形式: [..., seq_len, heads, dim/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 广播频率张量: [1, seq_len, 1, dim/2]
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])

    # 复数乘法完成旋转，还原为实数维度
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 施加 mscale 温度增益
    xq_out = xq_out.type_as(xq) * mscale
    xk_out = xk_out.type_as(xk) * mscale

    return xq_out, xk_out