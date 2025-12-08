'''
check_point：用于保存当前模型相关的信息
用途：
    (1)防止由于意外原因中断训练，导致需要重新训练，所以训练过程中保存相关参数、梯度、lr等等
    (2)用于分析模型的训练过程，比如保存损失这些内容
1. nn.module 以及 nn.Optim.Optimizer 都有 state_dict() 和 load_state_dict()的方法
2. torch.save(obj,dest)用于保存 obj 通常字典类型
'''
import torch



def save_checkpoint(
        model: torch.nn.Module,
        opimizer: torch.optim.Optimizer,
        iteration: int,
        out
):
    state_dict = {
        'epoch': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opimizer.state_dict()
    }
    torch.save(state_dict,out)

def load_checkpoint(
        src,
        model,
        optimizer
):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['epoch']