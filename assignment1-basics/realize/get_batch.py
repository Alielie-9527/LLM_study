import torch

def get_batch(data,batch_size,context_length,device):
    '''
    data为np.arryay格式
    从data中获取一个batch的数据
    data:torch.Tensor 一维张量，包含token的索引
    batch_size:int 每个batch的样本数量
    context_length:int 上下文长度（序列长度）
    device:torch.device 设备信息
    '''
    #随机产生batch_size个起点
    data = torch.tensor(data,dtype=torch.long)
    starts = torch.randint(0,len(data)-context_length,(batch_size,))
    # torch.stack拼接张量
    inputs = torch.stack([data[s:s+context_length] for s in starts]).to(device)
    targets = torch.stack([data[s+1:s+1+context_length] for s in starts]).to(device)
    return (inputs,targets)