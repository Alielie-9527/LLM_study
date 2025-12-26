# 大模型微调LoRA篇

## 一、LoRA原理

LoRA 的核心想法很简单：任意矩阵 $W$ 都可以用 SVD 分解 $A = U \Sigma V^T$（推导细节见另一篇笔记）。

其中 $U$ 和 $V^T$ 是正交矩阵，表示旋转；$\Sigma$ 负责在坐标轴方向缩放。

- 奇异值 $\sigma$ 决定拉伸倍数；$\sigma$ 越大说明这个方向越重要，越小/为 0 则说明不重要。

SVD 会将奇异值从大到小排序。保留前 $k$ 个奇异值后重构出的矩阵，是原矩阵在低维下的最佳低秩近似。

---

LoRA 论文的核心假设是“低内在维度”（Low Intrinsic Dimension）。

- 问题：全量微调时，权重变化量 $\Delta W$ 维度巨大（如 `4096×4096`）。
- SVD 视角：对理想的 $\Delta W$ 做 SVD，会发现只有少数奇异值很大，大部分接近 0，信息集中在少数方向。
- LoRA 做法：直接定义两个小矩阵 $A$ 和 $B$（秩为 $r$），强制 $\Delta W = BA$，让网络通过训练学出最佳的低秩近似。

因此有公式：
$$
\text{output} = (W_0 + \Delta W)x = (W_0 + BA)x
$$

![LoRA流程示意](assets/LoRA流程.png)

## 二、秩的选择策略

### 1. 权衡表示能力与效率

- 参数量：$\Delta W = BA$，可训练参数约为 $r(d+k)$。因为 $r \ll d,k$，引入的额外显存远小于全量微调。
- 近似能力：秩 $r$ 决定 $BA$ 的表达上限，过大会过拟合，过小会欠拟合。

### 2. 实用方法

1. 经验评估：尝试一组 $r$（常用 2 的幂）并在验证集上对比。
2. 计算预算：受 GPU 内存限制。
3. 性能饱和：观察 $r$ 增加到某点后是否开始过拟合。
4. 结合 $\alpha$：$\alpha$ 控制整体更新幅度，$r$ 控制结构容量，两者需联动调节。

## 三、缩放因子 $\dfrac{\alpha}{r}$

为什么要除以 $r$？这是一个归一化操作：A、B 初始化后，$r$ 越大，$BA$ 的输出幅度越大；除以 $r$ 可以让不同 $r$ 的数值范围相近，减少频繁调学习率的需求。

$\alpha$ 起到整体缩放作用：在归一化后乘以 $\alpha$ 相当于调节更新强度（类似调整学习率）。

一句话：$\tfrac{1}{r}$ 让你改 $r$ 时不用改学习率；$\alpha$ 让你在固定 $r$ 时也能调学习率。它们共同平衡 $W_0$ 的通用知识和 $\Delta W$ 的特定调整。

- 设定 $\alpha = r$：常见启发式，可得到简洁的 $h = W_0x + BAx$ 形式。
- 固定 $\alpha$：如 16/32/64，不随 $r$ 变，让 $\alpha$ 直接反映期望的调整强度。
- 将 $\alpha$ 视为独立超参：与 $r$、学习率一起做网格/随机/贝叶斯搜索。

## 四、手搓 LoRA 的线性层

先回顾公式（考虑转置）：

- 标准线性层：$y = x W_0^{T} + b,\; W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$。
- LoRA 线性层（冻结 $W_0$）：$y = x W_0^T + x (BA)^T \tfrac{\alpha}{r} + b = x W_0^T + x A^T B^T \tfrac{\alpha}{r} + b$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # lora 参数
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # 冻结主权重，只训练 LoRA 参数
            self.weight.requires_grad = False
        self.reset_parameters()

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # 初始化 LoRA 参数
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
      
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # LoRA 权重从主权重里剥离出来，保证训练时只更新 A 和 B，不更新主权重
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False # 表示当前模型融合过
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):

        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

if __name__ == '__main__':
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 定义参数
    in_features = 10
    out_features = 5
    r = 4
    lora_alpha = 8
    
    print("=== 开始测试 Linear LoRA 层 ===")
    
    # 1. 测试初始化
    layer = Linear(in_features, out_features, r=r, lora_alpha=lora_alpha)
    print(f"初始化检查: lora_A shape={layer.lora_A.shape}, lora_B shape={layer.lora_B.shape}")
    assert layer.lora_A.shape == (r, in_features)
    assert layer.lora_B.shape == (out_features, r)
    assert layer.weight.requires_grad == False, "主权重应该被冻结"
    assert layer.lora_A.requires_grad == True, "LoRA A 应该是可训练的"
    print("初始化测试通过 ✅")

    # 2. 测试前向传播 (初始状态)
    # 由于 lora_B 初始化为 0，初始输出应该与普通 Linear 层一致
    x = torch.randn(2, in_features)
    output = layer(x)
    expected = F.linear(x, layer.weight, layer.bias)
    assert torch.allclose(output, expected, atol=1e-6), "初始输出应该与普通 Linear 层一致"
    print("初始前向传播测试通过 ✅")

    # 3. 测试 LoRA 效果
    # 手动设置权重以验证计算逻辑
    nn.init.constant_(layer.lora_A, 1.0)
    nn.init.constant_(layer.lora_B, 1.0)
    # scaling = 8/4 = 2
    # LoRA term = (x @ A.T @ B.T) * scaling
    # 如果 x 全为 1 (1, 10)
    # x @ A.T (1, 4) -> 每个元素是 10 (1*1 * 10)
    # (x @ A.T) @ B.T (1, 5) -> 每个元素是 10 * 4 = 40
    # result * scaling (2) -> 80
    
    x_ones = torch.ones(1, in_features)
    output_ones = layer(x_ones)
    base_output = F.linear(x_ones, layer.weight, layer.bias)
    diff = output_ones - base_output
    
    expected_diff = 80.0
    assert torch.allclose(diff, torch.tensor(expected_diff), atol=1e-5), f"LoRA 增量计算错误, 期望 {expected_diff}, 实际 {diff.mean().item()}"
    print("LoRA 增量计算测试通过 ✅")

    # 4. 测试权重合并 (eval 模式)
    print("测试权重合并逻辑...")
    original_weight = layer.weight.data.clone()
    
    # 切换到 eval 模式 -> 应该触发合并
    layer.eval()
    assert layer.merged == True, "Eval 模式下 merged 标记应为 True"
    assert not torch.allclose(layer.weight.data, original_weight), "Eval 模式下主权重应该发生变化 (合并了 LoRA)"
    
    # 再次前向传播，结果应该保持一致
    output_eval = layer(x_ones)
    assert torch.allclose(output_eval, output_ones, atol=1e-6), "合并权重后的输出应该与未合并时一致"
    
    # 切换回 train 模式 -> 应该触发解绑
    layer.train()
    assert layer.merged == False, "Train 模式下 merged 标记应为 False"
    assert torch.allclose(layer.weight.data, original_weight, atol=1e-5), "Train 模式下主权重应该恢复原状"
    print("权重合并/解绑测试通过 ✅")
    
    print("=== 所有测试通过 ===")

        
```

