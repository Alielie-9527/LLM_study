import torch
from torch import nn

'''
激活函数SwiGLU的实现
FFN:
前向传播公式：
FFN(x) = W₂[SiLU(W₁x) ⊙ W₃x]

分步计算：
门控信号：g = W₁x ∈ ℝ<sup>d<sub>ff</sub></sup>
值信号：v = W₃x ∈ ℝ<sup>d<sub>ff</sub></sup>
激活门控：g̃ = SiLU(g) = g ⊙ σ(g)
门控相乘：h = g̃ ⊙ v
SiLU函数定义为SiLU(x) = x * sigmoid(x)
W₁x相当于传统的第一次线性变换，
W₂h相当于传统的第二次线性变换.
'''
class SwiGLU(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff=None,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}
        if d_ff is None:
            d_ff_ideal = (8/3) * d_model
            d_ff = int(d_ff_ideal//64)*64
         
        self.w1 = nn.Parameter(torch.empty((d_ff,d_model),**factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((d_model,d_ff),**factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff,d_model),**factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self,x):
        #门控信号
        gate = torch.matmul(x,self.w1.T)
        value = torch.matmul(x, self.w3.T)
       # SiLU(x) = x * sigmoid(x)，不是 sigmoid(x * sigmoid(x))!
        activated_gate = gate * torch.sigmoid(gate)  # 这就是SiLU
         # 门控相乘
        gated_value = activated_gate * value 
        output = torch.matmul(gated_value, self.w2.T) 
        return output