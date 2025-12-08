import torch
from torch import nn

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

        #self.register_buffer('inv_freq', inv_freq, persistent=False)

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


        
        # x0_rotated shape: [total_batch, seq_len, d_k//2]
        x0_rotated = x0 *cos -x1 *sin
        x1_rotated = x0 * sin +  x1 * cos

        x_rotated = torch.stack((x0_rotated,x1_rotated),dim=-1)
        x_rotated = x_rotated.view(total_batch, seq_len, d_k)
        x_rotated = x_rotated.view(orignal_shape)

        return x_rotated



