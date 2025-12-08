import torch
from torch import nn
import math


#测试通过吧   
class OptimizedLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 将 device 和 dtype 参数字典传递给张量构造函数
        # 这是最高效、最正确的做法。它能一步到位地在指定设备上创建指定类型的张量。
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features

        #  使用 torch.empty 替代 torch.zeros
        # 我们马上就要用随机值覆盖它，所以没必要先用0去填充内存，empty() 更快。
        # 遵循PyTorch命名约定，使用 'weight' (单数)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:

        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = torch.matmul(x, self.weight.T)
        return output

    # (可选) 添加 extra_repr 以便获得更好的打印输出
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'        # 偏置项被移除以简化实现
