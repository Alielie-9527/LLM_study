import torch
import torch.nn as nn
'''
rope的负数版本实现，参考原论文，以及ai帮助实现
总的来看，RoPE位置编码实际上相当于两个维度两个维度一组，然后进行旋转
这里实际上就相当于线性代数中的旋转矩阵 
重要公式：
（1）⟨q, k⟩ = Re[q · k*]  q=[q1,q2] = q1+q2i 
左侧是向量点积： q*k^T 
右侧则是将两个虚数相乘 将q和k看成 : q1+q2i 形式
Re则是取实数部分，结果是相同的。 
RoPE的核心思想就是:⟨f_q(x_m, m), f_k(x_n, n)⟩ = g(x_m, x_n, m-n)
找到满足这个方程的解.
（2） q_m = q · e^(imθ)    # query 在位置 m ，那么实际上就可以利用虚数计算出q_m 和k_n然后重新转为向量再进行点乘即可     
     k_n = k · e^(inθ)    # key 在位置 n
     实际上就是将q和k转成虚数表示，然后通过  · e^(imθ) ，实际上就是进行旋转 
（3）⟨q_m, k_n⟩ = Re[q·e^(imθ) · (k·e^(inθ))*]
          = Re[q·e^(imθ) · k*·e^(-inθ)] 
          = Re[q·k* · e^(i(m-n)θ)]
          = Re[(q·k*) · e^(i(m-n)θ)]
'''
class ComplexRotary_Embedding(nn.Module):
    def __init__(self,dim,max_seq_len=2048,base=10000):
        super().__init__()
        self.dim = dim  # dim 词嵌入维度 or 隐藏层维度
        inv_freq = 1.0/(base**torch.arange(0,dim,2).float()//dim) # 不同维度的θ
        self.register_buffer('inv_freq', inv_freq)  # 更高效

    def forward(self,x,position):  # position可以有更复杂的用法，目前没实操，推理过程的位置嵌入
        """
        参数:
            x: shape (batch, seq_len, dim) - 实数向量
            positions: shape (seq_len,)
        返回:
            rotated: shape (batch, seq_len, dim) - 旋转后的向量
        """

        batch_size, seq_len,dim = x.shape # 获取数入的维度，还未考虑多头
        x_complex = torch.view_as_complex(
            x.reshape(*x.shape[:-1],-1,2).contiguous()
        ) # 词嵌入维度，两两一组拆成 q1 + q2 i的形式
        # x_complex b,seq_len, dim//2
        # 计算各维度，各个位置需要旋转的角度
        freqs = position[:,None].float()*self.inv_freq[None,:] # (seqlen,dim//2)
        rotation_complex = torch.polar(
            torch.ones_like(freqs),
            freqs
        ) # 生成模长为1，角度freqs的 e^(iθ) = cos(θ) + i·sin(θ)

        # 复数乘法 实际上就是旋转
        rotation_complex = rotation_complex.unsqueeze(0) #  (1, seq_len, dim/2)
        x_rotated_complex = x_complex * rotation_complex # 广播

        # 转为实数
        x_rotated = torch.view_as_real(x_rotated_complex)  # (batch, seq_len, dim/2, 2)
        x_rotated = x_rotated.reshape(batch_size, seq_len, dim)

        return x_rotated


# 后续为验证和使用
def compute_attention_with_complex(q, k, rope, positions):
    """使用复数 RoPE 计算注意力分数"""

    # 应用 RoPE
    q_rot = rope(q, positions)  # (batch, seq_len, dim)
    k_rot = rope(k, positions)

    # 方法 1: 标准点积（实数）
    attn_real = q_rot @ k_rot.transpose(-2, -1)

    # 方法 2: 复数视角验证 Re[q·k*]
    batch_size, seq_len, dim = q.shape

    # 转换为复数
    q_complex = torch.view_as_complex(
        q_rot.reshape(batch_size, seq_len, -1, 2).contiguous()
    )
    k_complex = torch.view_as_complex(
        k_rot.reshape(batch_size, seq_len, -1, 2).contiguous()
    )

    # 计算 q·k* (复数点积)
    # einsum: 'bqd,bkd->bqk' where d is complex dimension
    attn_complex = torch.einsum('bqd,bkd->bqk', q_complex, k_complex.conj())

    # 取实部
    attn_from_complex = attn_complex.real

    return attn_real, attn_from_complex