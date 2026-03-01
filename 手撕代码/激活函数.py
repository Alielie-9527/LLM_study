#各种激活函数的实现
import torch
import torch.nn.functional as F

# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Tanh 激活函数
def tanh(x):
    return torch.tanh(x)

# ReLU 激活函数
def relu(x):
    # 自己实现
    return torch.where(x > 0, x, torch.zeros_like(x))

# leaky ReLU 激活函数
def leaky_relu(x, negative_slope=0.01):
    return torch.where(x > 0, x, negative_slope * x)
x = torch.tensor([-1.0, 0.0, 1.0])
print("Leaky ReLU:", leaky_relu(x))

# siLU实现
def silu(x):
    return x * sigmoid(x)

# swiGLU实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        # 为了保持参数量与标准 Transformer 相当，hidden_dim 通常是 2/3 * 4d
        # multiple_of 是为了硬件对齐加速
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # W_g: 门控投影
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # W_v: 值投影
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # W_out: 输出投影
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # 1. 计算 Gate 分支: Swish(xW_g) -> F.silu(self.w1(x))
        # 2. 计算 Value 分支: xW_v -> self.w3(x)
        # 3. 逐元素相乘: Gate * Value
        # 4. 输出投影: * W_out
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

