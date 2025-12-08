import torch
from torch import nn

#测试通过
'''
1.相较于原来的LayerNorm，减小了计算开销，需要两次遍历，一次算均值，一次算方差，同时需要存储bias（占用内存空间）
2.RMS的核心思想是**去除均值中心化步骤**，只保留方差归一化，从而简化计算。
'''
'''
对比：
（1）LayerNorm:
跨特征维度进行归一化，计算均值和方差时考虑所有特征
（2）RMSNorm:（layerNorm的简化版）
对每个token的所有特征进行归一化，独立计算均值和方差
（3）BatchNorm:
对一个batch的中，相同通道的特征进行归一化，跨样本计算均值和方差
'''

'''
因为我们希望模型学习到一种通用的、与位置无关的特征缩放规则
这也是为什么gain参数是与d_model维度相同的向量，而不是与序列长度或批次大小相关的参数
'''
class RMSnorm(nn.Module):
    def __init__(self,d_model:int, eps:float=1e-6,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}
        self.d_model = d_model
        # 用于放缩、初始化为1，表示刚开始只进行归一化
        self.gain = nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.eps = eps


    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # 防止计算溢出
        #对于 batch 中的每一个 token，独立地计算其 d_model 维特征的均方值
        rms = torch.sqrt(torch.mean(x**2,dim =-1,keepdim=True)+self.eps)   # shape (batch_size,seq_len,1)
        result = x / rms *self.gain
        return result.to(in_dtype)