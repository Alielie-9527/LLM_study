import torch
from torch import nn

'''
softmax实现
'''

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



