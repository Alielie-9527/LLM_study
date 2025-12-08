import math

'''
用于实现余弦退火算法：
参数：
当前迭代轮次  最大学习率   最小学习率   热身迭代次数  退火迭代次数
t           αmax       αmin       Tw         Tc   
三个阶段：
1.学习率预热阶段，学习率慢慢增长到 αmax(最大学习率)
    if t<Tw
    特点：线性增长
    公式：αt = (t / Tw) * αmax  
2.余弦退火阶段：
    if Tw ≤ t ≤ Tc
    特点：平滑衰减
    公式： αt = αmin + (1/2) * (1 + cos(π * (t - Tw) / (Tc - Tw))) * (αmax - αmin)
3.退火后：
    保持最小学习率
'''

def Cosine_annealing(
        t:int,
        max_lr:int,
        min_lr:int,
        warmup_iters:int,
        cosine_cycle_iters: int
):
    if t< warmup_iters:
        lr = (t/warmup_iters) * max_lr
    elif  warmup_iters<=t<cosine_cycle_iters:
        lr = min_lr + 0.5*(1+math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_lr - min_lr)
    else:
        lr = min_lr
    return lr
