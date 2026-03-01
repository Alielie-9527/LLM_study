from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self,dim:int, eps:float=1e-5):
        super().__init__()
        self.eps =eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
    def forward(self,x):
        # x.float()是确保x的数据类型为float，以避免在计算过程中出现数据类型不匹配的问题。最后再将结果转换回x的原始数据类型。
        # 这种做法在某些情况下是必要的，特别是当输入数据可能是半精度（float16）或其他较低精度的数据类型时，使用float32进行计算可以提高数值稳定性和精度。最后再将结果转换回原始数据类型，以保持与输入数据的一致性。
        return self.weight *self._norm(x.float()).type_as(x)