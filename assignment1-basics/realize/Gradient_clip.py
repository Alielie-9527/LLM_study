'''
梯度裁剪：
在深度学习训练过程中，梯度裁剪（Gradient Clipping）是一种常用的技术，用于防止梯度爆炸问题。
梯度爆炸会导致模型参数更新过大，从而使得训练过程不稳定，甚至无法收敛。通过梯度裁剪，可以将梯度的范数限制在一个预设的阈值之内，从而保持训练的稳定性。
'''

'''
params.grad 是一个对象
而params.grad.data 是一个tensor
'''

def gradient_clip(params, max_norm, eps=1e-6):
    """
    更简洁的梯度裁剪实现
    """
    # 方法1：使用平方和再开方（标准做法）
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            # 直接计算平方和，避免多次 .item() 转换
            # detach() 防止梯度追踪 节省内存
            total_norm += p.grad.detach().data.pow(2).sum().item()
    total_norm = total_norm ** 0.5  # 最后统一开方

    # 方法2：使用PyTorch的norm函数（更简洁）
    # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]), 2)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

