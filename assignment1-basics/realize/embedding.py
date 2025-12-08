import torch
from torch import nn

#测试通过


'''
嵌入层的工作方式实际上是进行查表， 接受 （batch_size,seq_len) 然后嵌入矩阵（vocab_size,d_moedl)
N(0,1) truncated at [-3, 3]
'''


class MyEmbedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 实际上就是查找表
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self)->None:
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3,b=3)

    def forward(self,token_ids:torch.Tensor):
        # token_ids (batch_size,seq_len)  先展平后查表
        orignal_shape = token_ids.shape

        flat_input = token_ids.view(-1)

        flat_output = self.weight[flat_input]

        output = flat_output.view(*orignal_shape,self.embedding_dim)
        return output

