import torch
import torch.nn as nn
from typing import Optional, Tuple

from numpy import dtype
from sympy import false
from torch.cuda import device
from torch.utils.hipify.hipify_python import value


class LLaMARotaryEmbedding(nn.Module):
    """
    LLaMA 风格的 RoPE 实现
    特点:
      1. 预计算 cos/sin 缓存
      2. Interleaved 方式（交错）
      3. 支持动态序列长度
      **而在 LLaMA 的实现中，特征对是向量的前后两半
      采用的加法计算过程
    """
    def __init__(
            self,
            dim:int, # 一定要指定int
            max_position_embeddings:int =2048,
            base:int= 10000,
            device:Optional[torch.device]=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings # 最大句子长度
        self.base = base
        #计算 inv_freq
        # 公式: inv_freq_i = 1.0 / (base^(2i/dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0,self.dim,2 , dtype=torch.float32) / self.dim))
        self.register_buffer('inv_freq',inv_freq,persistent=False)
        self._set_cos_sin_cache(
            seq_len = max_position_embeddings,
            device = device,
            dtype = torch.get_default_dtype()
        )

    # 提前计算好 计算 sin 和 cos 的cache
    def _set_cos_sin_cache(self,seq_len:int, device, dtype):
        self.max_seq_len_cached = seq_len
        # 创建位置索引  imθ 向量实际旋转的角度 由 i （词在句子中的位置）和 m（第几个维度）决定
        t = torch.arange(self.max_seq_len_cached,device=device,dtype=torch.float32)

        # 旋转矩阵
        freqs = t.unsqueeze(-1) *self.inv_freq.unsqueeze(0)  # 外积公式也可以 freqs = torch.outer(t,self.inv_freq)

        # 拼接
        emb = torch.cat((freqs,freqs),dim=-1)  # 因为 这个架构将 0和n 两个一组，

        #计算cos 和 sin
        self.register_buffer('cos_cached',emb.cos().to(dtype),persistent=False) #persistent = False 表示不存到模型中
        self.register_buffer('sin_cached',emb.sin().to(dtype),persistent=False)

    def forward(self,x:torch.Tensor,seq_len:Optional[int]=None):
        """
        参数:
            x: (batch, num_heads, seq_len, head_dim)
            seq_len: 序列长度（可选）

        返回:
            cos, sin: (1, 1, seq_len, head_dim)
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len >self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len,device = x.device,dtype=x.dtype)


        # 取前 seq_len 行，即对应当前输入序列长度的位置编码。
        # .to(dtype=x.dtype) 确保返回的类型与输入 x 一致，支持混合精度训练。
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )



'''
[x_m,x_m+1]进行旋转得到的 [x'_m, x'_m+1]
x'_m = x_m * cos(pos * θ_i) - x_{m+1} * sin(pos * θ_i)
x'_{m+1} = x_{m+1} * cos(pos * θ_i) + x_m * sin(pos * θ_i)
'''

def rotate_half(x:torch.Tensor)->torch.Tensor:
    x1 = x[...,:x.shape[-1]//2]  # ...表示所有剩余维度
    x2 = x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码（LLaMA 风格）

    参数:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) 或 (1, 1, seq_len, head_dim)
        sin: (seq_len, head_dim) 或 (1, 1, seq_len, head_dim)
        position_ids: (batch, seq_len) 可选，用于不连续位置

    返回:
        q_embed, k_embed: 应用 RoPE 后的 q 和 k
    """
    # 如果提供了 position_ids，使用它来索引
    if position_ids is not None:
        # 广播到正确的形状
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # 使用连续位置
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # 应用旋转：x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

'''
下面实现的是自注意力，用于transformer编码器，
解码器的结构不一样，因为 q  （k,v) 来源分别都不一样
'''
class LLaMAAttention(nn.Module):
    """LLaMA 风格的多头注意力（包含 RoPE）"""

    def __init__(
            self,
            hidden_size:int =4096,
            num_heads:int =32,
            max_position_embedding:int =2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q,K,V 投影
        self.q_proj = nn.Linear(hidden_size,hidden_size,bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # RoPE配置
        self.rotary_emb = LLaMARotaryEmbedding(
            self.head_dim,
            max_position_embeddings = max_position_embedding
        )

    def forward(
            self,
            hidden_states:torch.Tensor,
            attention_mask:Optional[torch.Tensor] = None,
            position_ids:Optional[torch.tensor] = None
    )->torch.Tensor:

        """
        参数:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, seq_len, seq_len) 可选
            position_ids: (batch, seq_len) 可选
        """
        batch_size,seq_len,_ = hidden_states.shape

        # 投影 QKV矩阵
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 多头调整
        query_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size,seq_len,self.nums_heads,self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用Rope
        cos ,sin = self.rotary_emb(value_states,seq_len= seq_len)
        query_states,key_states= apply_rotary_pos_emb(
            query_states,key_states,cos,sin,position_ids
        )

        # 计算注意力
        attn_weights = torch.matmul(query_states,key_states.transpose(-2,-1))/ (self.head_dim ** 0.5 )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights,dim = -1)

        # 加权V
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape(batch_size,seq_len,self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output



