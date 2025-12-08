import torch
'''
该文件用于实现交叉熵损失函数
给出：
predictions: [batch_size, num_classes] 经过softmax的预测概率分布
targets: [batch_size] 真实类别标签
返回：
loss: 标量交叉熵损失
'''
'''
原理可以参考极大似然估计：
p(x|y) 在已经知道x的情况下，最大化 y 的似然函数
交叉熵等价于极大似然估计
'''
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    手动实现交叉熵损失函数
    logits: [batch_size, num_classes] 模型输出
    targets: [batch_size] 真实类别标签
    返回：
    loss: 标量交叉熵损失
    这里采取log-sum-exp技巧，防止数值不稳定
    '''
    # 计算log-sum-exp
    max_logits, _ = torch.max(logits,dim=-1,keepdim=True) # [batch_size,1]
    logits_stable = logits -max_logits # 数值稳定

    exp_logits = torch.exp(logits_stable)
    sum_exp_logits = torch.sum(exp_logits,dim=-1,keepdim=True) # [batch_size,1] 指定维度
    log_sum_exp_logits = torch.log(sum_exp_logits).squeeze() + max_logits.squeeze()
    target_logits = logits[torch.arange(logits.size(0)),targets]
    loss = log_sum_exp_logits - target_logits
    return loss.mean()

