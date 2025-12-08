from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

'''
AdamW 将 权重衰减（防止参数过大而在l += 1/2 ||λ||**2）
与梯度更新解耦
1.只使用原始梯度（不含 L2 正则化项）来计算 Adam 的自适应更新步骤（包括动量和 RMSprop）
2.只使用原始梯度（不含 L2 正则化项）来计算 Adam 的自适应更新步骤（包括动量和 RMSprop）
好处：解耦 (Decoupled)。衰减效果独立于梯度，对所有权重施加一个与其大小成正比的、稳定的衰减力。
原处理：耦合 (Coupled)。衰减效果受到梯度历史（二阶矩 v_t）的影响，不稳定且效果减弱。
'''

'''
偏差矫正的原因，初始化动量矩阵为0，则导致在训练初期，矩估计值被严重低估了！这是因为初始的零值需要很多步才能被"遗忘"。
'''


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
        self.step_count = 0

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # 获取或初始化状态
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                # 更新步数
                state['step'] += 1

                # 更新一阶矩和二阶矩，原地操作，节省内存
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正（关键修正） 不能直接修改state['m']的状态
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 计算校正后的矩估计
                m_hat = state['m'] / bias_correction1
                v_hat = state['v'] / bias_correction2


                # 权重衰减（与梯度更新解耦）
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Adam更新：使用校正后的矩估计
                denom = v_hat.sqrt().add_(eps)
                p.data.addcdiv_(m_hat, denom, value=-lr)

        return loss
    
