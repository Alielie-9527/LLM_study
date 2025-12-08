import math
from typing import Optional
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

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

#通过预计算的sin和cos实现RoPE
'''
RoPE的实现
'''
class myRoPE(nn.Module):
    def __init__(self,d_k,theta,max_seq_len=512):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 这是 对某个 q 或 k中不同维度两两一组进行旋转
        inv_freq = 1.0/(theta ** (torch.arange(0,d_k,2).float()/d_k))

        self.register_buffer('inv_freq', inv_freq, persistent=False)

        #这个是对位于句子中不同位置的词进行旋转(seq_len,d_k/2)
        position = torch.arange(max_seq_len)
        freq = position.unsqueeze(-1) * inv_freq.unsqueeze(0) # 广播

        self.register_buffer('sin_cached',freq.sin(),persistent=False) #不随模型保存
        self.register_buffer('cos_cached',freq.cos(),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)-> torch.Tensor:
        '''
        x的形状(**,seq_len,d_k)
        position(**,seq_len)
        '''
        orignal_shape = x.shape
        seq_len = orignal_shape[-2]
        d_k = orignal_shape[-1]

        # 模型鲁棒性
        assert d_k == self.d_k, f"输入维度{d_k}与RoPE维度{self.dim}不匹配"
       

        # 展平便于处理
        x_flat = x.view(-1,seq_len,d_k)
        token_positions_flat = token_positions.view(-1,seq_len)
        total_batch = x_flat.shape[0]

        x_reshaped = x_flat.view(total_batch,seq_len,d_k//2,2) #  列向量变为行向量：[x1,x2]

        x0 = x_reshaped[..., 0]  # [total_batch, seq_len, d_k//2]
        x1 = x_reshaped[..., 1]  # [total_batch, seq_len, d_k//2]

        # 获取位置索引，确保不越界
        positions = token_positions_flat.long().clamp(0, self.max_seq_len - 1)  # [seq_len]

        cos = self.cos_cached[positions]  # [seq_len,dk//2]
        sin = self.sin_cached[positions]

        x0_rotated = x0 *cos -x1 *sin
        x1_rotated = x0 * sin +  x1 * cos

        x_rotated = torch.stack((x0_rotated,x1_rotated),dim=-1)
        x_rotated = x_rotated.view(total_batch, seq_len, d_k)
        x_rotated = x_rotated.view(orignal_shape)

        return x_rotated



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
    attn_score = mySoftmax(score,dim=-1)
    output = torch.matmul(attn_score,V)
    return output



class MultiHeadAttention_RoPE(nn.Module):
    def __init__(self,d_model,num_heads,max_seq_len=512,theta=10000):
        super().__init__()
        self.d_model =d_model
        self.num_heads = num_heads
        self.d_k = int(d_model // num_heads)
        self.q_proj = nn.Parameter(torch.empty(d_model,d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(self.d_model,d_model))
        self.rope = myRoPE(self.d_k,theta,max_seq_len)

        # 可添加一个参数初始化
    def forward(self,in_features,token_positions=None):
        seq_len = in_features.shape[-2]
        Q = torch.matmul(in_features,self.q_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2).contiguous()
        K = torch.matmul(in_features,self.k_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2).contiguous()
        V = torch.matmul(in_features,self.v_proj.T).view(-1,seq_len,self.num_heads,self.d_k).transpose(1,2)

    
        if token_positions is not None:
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
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

