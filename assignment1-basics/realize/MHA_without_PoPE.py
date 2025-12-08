import math
from typing import Optional
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn



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
        score = score.masked_fill(mask,float('-inf'))  # 错误做法导致trues replaced by -inf
        # score = score.masked_fill(mask == False, float('-inf'))
    attn_score = torch.softmax(score,dim=-1)
    output = torch.matmul(attn_score,V)
    return output



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model =d_model
        self.num_heads = num_heads
        self.d_k = int(d_model // num_heads)
        self.q_proj = nn.Parameter(torch.empty(d_model,d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(self.d_model,d_model))

        # 可添加一个参数初始化
    def forward(self,in_features):
        seq_len = in_features.shape[-2]
        Q = torch.matmul(in_features,self.q_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K = torch.matmul(in_features,self.k_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V = torch.matmul(in_features,self.v_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2) 

        # 因果掩码矩阵
        casual_mask = torch.triu(torch.ones((seq_len,seq_len),dtype=torch.bool),diagonal=1).unsqueeze(0)
        output = scaled_dot_product_attention(Q,K,V,casual_mask)
        output = output.transpose(1,2).contiguous().view(-1,seq_len,self.d_model)
        # attn_scores = torch.matmul(Q,K.transpose(-1,-2))
        # attn_scores = attn_scores.masked_fill(casual_mask,float('-inf'))
        # attn_scores = self.softmax(attn_scores)
        #
        # output = torch.matmul(attn_scores,V.transpose(-1,-2))
        output_proj = torch.matmul(output,self.o_proj.T)
        return output_proj

