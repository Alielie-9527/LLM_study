import math
from typing import Optional

import torch
from torch import nn, Tensor


#要求使用ROPE
#使用RMSNorm归一化

def mySoftmax(x:torch.Tensor,dim):
    """
        手动实现 softmax 函数

        Args:
            x: 输入张量
            dim: 计算 softmax 的维度

        Returns:
            经过 softmax 变换后的张量
        """
    #数值稳定性，减去最大值
    x_max = torch.max(x,dim=dim,keepdim=True)[0]
    '''
    该函数返回的是元组，最大值 和 序号
    '''
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x,dim=dim,keepdim=True)
    softmax_result = exp_x / sum_exp
    
    return softmax_result

class RMSnorm(nn.Module):
    def __init__(self,d_model:int, eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}
        self.d_model = d_model
        # 用于放缩、初始化为1，表示刚开始只进行归一化
        self.gain = nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.eps = eps


    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # 防止计算溢出
        # self.eps要在里面，否则是错误的
        rms = torch.sqrt(torch.mean(x**2,dim =-1,keepdim=True)+self.eps)
        result = x / rms *self.gain
        return result.to(in_dtype)

class myRoPE(nn.Module):
    def __init__(self, d_k, theta, max_seq_len=512):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 这是 对某个 q 或 k中不同维度两两一组进行旋转
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 这个是对位于句子中不同位置的词进行旋转(seq_len,d_k/2)
        position = torch.arange(max_seq_len)
        freq = position.unsqueeze(-1) * inv_freq.unsqueeze(0)  # 广播

        self.register_buffer('sin_cached', freq.sin(), persistent=False)  # 不随模型保存
        self.register_buffer('cos_cached', freq.cos(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        x的形状(**,seq_len,d_k)
        position(**,seq_len)
        '''
        orignal_shape = x.shape
        seq_len = orignal_shape[-2]
        d_k = orignal_shape[-1]

        # 模型鲁棒性
        assert d_k == self.d_k, f"输入维度{d_k}与RoPE维度{self.d_k}不匹配"

        # 展平便于处理
        x_flat = x.view(-1, seq_len, d_k)
        token_positions_flat = token_positions.view(-1, seq_len)
        total_batch = x_flat.shape[0]

        x_reshaped = x_flat.view(total_batch, seq_len, d_k // 2, 2)  # 列向量变为行向量：[x1,x2]

        x0 = x_reshaped[..., 0]  # [total_batch, seq_len, d_k//2]
        x1 = x_reshaped[..., 1]  # [total_batch, seq_len, d_k//2]

        # 获取位置索引，确保不越界
        positions = token_positions_flat.long().clamp(0, self.max_seq_len - 1)  # [seq_len]

        cos = self.cos_cached[positions]  # [seq_len,dk//2]
        sin = self.sin_cached[positions]

        # x0_rotated shape: [total_batch, seq_len, d_k//2]
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos

        x_rotated = torch.stack((x0_rotated, x1_rotated), dim=-1)
        x_rotated = x_rotated.view(total_batch, seq_len, d_k)
        x_rotated = x_rotated.view(orignal_shape)

        return x_rotated


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff=None,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if d_ff is None:
            d_ff_ideal = (8 / 3) * d_model
            d_ff = int(d_ff_ideal // 64) * 64

        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x):
        # 门控信号
        gate = torch.matmul(x, self.w1.T)
        value = torch.matmul(x, self.w3.T)
        # SiLU(x) = x * sigmoid(x)，不是 sigmoid(x * sigmoid(x))!
        activated_gate = gate * torch.sigmoid(gate)  # 这就是SiLU
        # 门控相乘
        gated_value = activated_gate * value
        output = torch.matmul(gated_value, self.w2.T)
        return output


def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
) -> Tensor:
    d_k = Q.shape[-1]  # 这里获得的是标量，用torch.sqrt会报错
    score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask, float('-inf'))  # 错误做法导致trues replaced by -inf
        # score = score.masked_fill(mask == False, float('-inf'))
    attn_score = mySoftmax(score, dim=-1)
    output = torch.matmul(attn_score, V)
    return output


class MultiHeadAttention_RoPE(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 max_seq_len=512,
                 theta=10000.0
    )->None:
        """
            初始化带有旋转位置编码的Transformer层。

            参数说明:
                d_model: 模型维度
                    - 决定词嵌入和隐藏状态的维度大小
                    - 必须是num_heads的整数倍
                    - 示例: 512, 768, 1024

                num_heads: 注意力头数量
                    - 将注意力机制并行化到多个子空间
                    - 约束: d_model % num_heads == 0
                    - 示例: 8, 12, 16

                d_ff: 前馈网络维度
                    - 前馈神经网络中间层的神经元数量

                weights: 权重参数张量
                    - 预训练模型的权重参数
                    - 形状: 应符合各层的参数形状要求
                    - 注意: 需要与模型架构完全匹配

                max_seq_len: 最大序列长度
                    - 模型支持处理的最大token数量
                    - 影响位置编码表的预计算大小
                    - 示例: 512, 1024, 2048

                theta: 旋转位置编码基础频率
                    - RoPE公式中的θ参数，控制波长
                    - 默认值: 10000.0 (原始论文使用)
                    - 较小的值提供更长的波长

            异常:
                如果d_model不能被num_heads整除，抛出ValueError
            """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = int(d_model // num_heads)
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(self.d_model, d_model))
        self.rope = myRoPE(self.d_k, theta, max_seq_len) #这种每个模块单独初始化并不是很好

        # 可添加一个参数初始化

    def forward(self, in_features, token_positions=None):
        seq_len = in_features.shape[-2]
        Q = torch.matmul(in_features, self.q_proj.T).view(-1, seq_len, self.num_heads, self.d_k).transpose(1,
                                                                                                           2).contiguous()
        K = torch.matmul(in_features, self.k_proj.T).view(-1, seq_len, self.num_heads, self.d_k).transpose(1,
                                                                                                           2).contiguous()
        V = torch.matmul(in_features, self.v_proj.T).view(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # 因果掩码矩阵
        casual_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).unsqueeze(0)
        output = scaled_dot_product_attention(Q, K, V, casual_mask)
        output = output.transpose(1, 2).contiguous().view(-1, seq_len, self.d_model)
        # attn_scores = torch.matmul(Q,K.transpose(-1,-2))
        # attn_scores = attn_scores.masked_fill(casual_mask,float('-inf'))
        # attn_scores = self.softmax(attn_scores)
        #
        # output = torch.matmul(attn_scores,V.transpose(-1,-2))
        output_proj = torch.matmul(output, self.o_proj.T)
        return output_proj

class Transform_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta):
        super().__init__()
        self.rmsnorm_1 = RMSnorm(d_model) #pre_norm
        self.rmsnorm_2 = RMSnorm(d_model) # pre_norm
        self.FNN =SwiGLU(d_model,d_ff) # 全连接层
        self.attention = MultiHeadAttention_RoPE(d_model,num_heads,max_seq_len,theta) #多头注意力

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        output = self.rmsnorm_1(x)
        output = self.attention(output, token_positions)
        output = output + x # 残差连接

        output_rms2 = self.rmsnorm_2(output)
        output_ffn = self.FNN(output_rms2)
        return output_ffn+output



