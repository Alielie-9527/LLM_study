import math
from typing import Optional
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

'''
该文件用于实现点积注意力机制
给出：
Q:[batch_size,...,seq_len,d_k]
k:[batch_size,...,seq_len,d_k]
v:[batch_size,...,seq_len,d_v]
attention_mask:bool[Tensor,"... seq_len,seq_len" | None

'''
def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor, 
        V: Tensor,
        mask: Optional[Tensor] = None,
) -> Tensor:
    d_k = Q.shape[-1]  # 这里获得的是标量，用torch.sqrt会报错
    score = torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d_k)
    if mask is not None:
        # score = score.masked_fill(mask,float('-inf')) 错误做法导致trues replaced by -inf
        score = score.masked_fill(mask == False, float('-inf'))
    attn_score = torch.softmax(score,dim=-1)
    output = torch.matmul(attn_score,V)
    return output


