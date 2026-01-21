import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(object):
    """
    稀疏分发器：用于辅助实现专家混合模型 (MoE)。
    
    主要功能：
    1. dispatch (分发): 将输入数据根据门控权重分发给对应的专家。
    2. combine (组合): 将各个专家计算后的结果根据门控权重进行加权组合。
    
    工作原理：
    类初始化时接收一个 `gates` 张量，shape为 [batch_size, num_experts]。
    只有当 gates[b, e] > 0 时，第 b 个样本才会分发给专家 e。
    
    使用示例:
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs) 
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    """

    def __init__(self, num_experts, gates):
        """初始化 SparseDispatcher."""
        print(f"初始化分发器，gates形状: {gates.shape}")
        print(f"gates张量:\n{gates}")
        # 示例: gates = tensor([[0.1, 0.9],      # 样本0: 专家0权重0.1, 专家1权重0.9
        #                      [0.8, 0.2],      # 样本1: 专家0权重0.8, 专家1权重0.2  
        #                      [0.0, 1.0]])     # 样本2: 专家0权重0.0, 专家1权重1.0

        self._gates = gates  # [batch_size, num_experts]
        self._num_experts = num_experts
        
        # 1. 找出所有非零门控值的索引 (即哪些样本选中了哪些专家)
        # gates: [batch_size, num_experts] -> nonzero_indices: [num_nonzero, 2] (样本索引, 专家索引)
        nonzero_indices = torch.nonzero(gates)
        print(f"非零元素位置:\n{nonzero_indices}")  # tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]])
        # 表示: [样本0,专家0], [样本0,专家1], [样本1,专家0], [样本1,专家1], [样本2,专家1]
        
        # sort(0)按列排序，得到 sorted_experts (值) 和 index_sorted_experts (原位置索引)
        # sorted_experts: [num_nonzero, 2], index_sorted_experts: [num_nonzero, 1]
        sorted_experts, index_sorted_experts = nonzero_indices.sort(0)
        print(f"排序后位置:\n{sorted_experts}")    # 按列排序后的坐标
        print(f"排序索引:\n{index_sorted_experts}") # 排序后元素在原数组中的位置
        # sorted_experts: tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]])
        # index_sorted_experts: tensor([[0], [2], [1], [3], [4]])
        # 排序规则: 先按第0列(样本索引)排序，再按第1列(专家索引)排序
        
        # 2. 分离出专家索引 (第2列)
        # sample_idx: [num_nonzero, 1], expert_idx: [num_nonzero, 1]
        sample_idx, expert_idx = sorted_experts.split(1, dim=1)
        self._expert_index = expert_idx  # [num_nonzero, 1]
        print(f"专家索引:\n{self._expert_index}")
        
        # 3. 获取对应的样本索引
        # index_sorted_experts[:, 1]: [num_nonzero] -> torch.nonzero(gates)[..., 0]: [num_nonzero] -> self._batch_index: [num_nonzero]
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        print(f"批次索引:\n{self._batch_index}")  # tensor([0, 1, 0, 1, 2])
        # 含义: 排序后的5个非零元素分别属于样本0,1,0,1,2
        
        # 4. 计算每个专家被分配到的样本数量
        # gates: [batch_size, num_experts] -> _part_sizes: [num_experts]
        self._part_sizes = (gates > 0).sum(0).tolist()
        print(f"每个专家分配的样本数: {self._part_sizes}")  # [2, 3] - 专家0分配2个样本，专家1分配3个样本
        # 专家0: 样本0,1 (gates[0,0], gates[1,0] 非零)
        # 专家1: 样本0,1,2 (gates[0,1], gates[1,1], gates[2,1] 非零)
        
        # 5. 为了后续 combine 时的加权，这里收集对应的非零权重值
        # gates: [batch_size, num_experts], _batch_index: [num_nonzero] -> gates_exp: [num_nonzero, num_experts]
        gates_exp = gates[self._batch_index.flatten()]
        print(f"根据样本索引取出的门控行:\n{gates_exp}")
        # gates_exp[0] = gates[0] = [0.1, 0.9]  (样本0的门控)
        # gates_exp[1] = gates[1] = [0.8, 0.2]  (样本1的门控) 
        # gates_exp[0] = gates[0] = [0.1, 0.9]  (样本0的门控)
        # gates_exp[1] = gates[1] = [0.8, 0.2]  (样本1的门控)
        # gates_exp[2] = gates[2] = [0.0, 1.0]  (样本2的门控)
        
        # gates_exp: [num_nonzero, num_experts], _expert_index: [num_nonzero, 1] -> _nonzero_gates: [num_nonzero, 1]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        print(f"非零门控权重:\n{self._nonzero_gates}")  # tensor([[0.1], [0.2], [0.9], [0.8], [1.0]])
        # 按排序顺序: [样本0专家0], [样本1专家1], [样本0专家1], [样本1专家0], [样本2专家1]

    def dispatch(self, inp):
        """
        分发输入给各个专家。
        
        参数:
          inp: 输入张量 [batch_size, <extra_input_dims>] 即 [batch_size, input_size]
        
        返回:
          一个包含 num_experts 个张量的列表。
          第 i 个张量的形状为 [expert_batch_size_i, <extra_input_dims>]。
        """
        print(f"\n开始dispatch，输入形状: {inp.shape}")
        print(f"输入数据:\n{inp}")
        # 示例: inp = tensor([[10, 11],    # 样本0
        #                     [20, 21],    # 样本1  
        #                     [30, 31]])   # 样本2
        
        # 根据 _batch_index 选出需要被处理的样本，相当于复制/重排
        # inp: [batch_size, input_size], _batch_index: [total_nonzero] -> inp_exp: [total_nonzero, input_size]
        inp_exp = inp[self._batch_index].squeeze(1)
        print(f"按批次索引重排后的输入:\n{inp_exp}")
        # inp_exp = tensor([[10, 11],    # 样本0
        #                   [20, 21],    # 样本1
        #                   [10, 11],    # 样本0 (重复)
        #                   [20, 21],    # 样本1 (重复)
        #                   [30, 31]])   # 样本2
        # 总共5个输入，对应5个非零门控
        
        # 根据每个专家分配到的样本数 (_part_sizes) 切分数据
        # result: 列表，每个元素 [part_size, input_size]
        result = torch.split(inp_exp, self._part_sizes, dim=0)
        print(f"切分后的专家输入:")
        for i, expert_inp in enumerate(result):
            print(f"  专家{i}: {expert_inp.shape} -> {expert_inp}")
        # 专家0: [2, 2] -> tensor([[10, 11], [20, 21]])  # 对应[样本0专家0, 样本1专家0]
        # 专家1: [3, 2] -> tensor([[10, 11], [20, 21], [30, 31]])  # 对应[样本0专家1, 样本1专家1, 样本2专家1]
        
        return result

    def combine(self, expert_out, multiply_by_gates=True):
        """
        组合各个专家的输出。
        
        参数:
          expert_out: 包含 num_experts 个张量的列表，每个 [expert_batch_size_i, output_size]
          multiply_by_gates: 是否乘上门控权重（通常为True）。
        
        返回:
          组合后的输出张量 [batch_size, output_size]。
        """
        print(f"\n开始combine，专家输出数量: {len(expert_out)}")
        # 示例: expert_out[0] = tensor([[100, 101], [200, 201]])  # 专家0输出
        #      expert_out[1] = tensor([[300, 301], [400, 401], [500, 501]])  # 专家1输出
        
        # 将列表拼接成一个大张量
        # expert_out: 列表 [expert_batch_size_i, output_size] -> stitched: [total_nonzero, output_size]
        stitched = torch.cat(expert_out, 0)
        print(f"拼接后的专家输出:\n{stitched}")
        # stitched = tensor([[100, 101], [200, 201], [300, 301], [400, 401], [500, 501]])
        
        if multiply_by_gates:
            print(f"原始专家输出权重:\n{self._nonzero_gates}")
            # stitched: [total_nonzero, output_size], _nonzero_gates: [total_nonzero, 1] -> 逐元素相乘
            stitched = stitched.mul(self._nonzero_gates)
            print(f"加权后的专家输出:\n{stitched}")
            # stitched = tensor([[10, 10.1], [40, 40.2], [270, 270.9], [320, 320.8], [500, 501]])
            # 权重: [0.1, 0.2, 0.9, 0.8, 1.0] 对应 [样本0专家0, 样本1专家1, 样本0专家1, 样本1专家0, 样本2专家1]
            
        # 创建零张量用于累加
        # zeros: [batch_size, output_size]
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        print(f"创建零张量形状: {zeros.shape}")
        
        # 根据 _batch_index 将结果累加回原来的 batch 位置
        # index_add(dim, index, source): 将 source 的行按 index 加到 self 的 dim 维度上
        # 这里处理了同一个样本被多个专家选中的情况（累加）
        # combined: [batch_size, output_size]
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        print(f"最终组合结果:\n{combined}")
        # _batch_index = [0, 1, 0, 1, 2]
        # stitched = [[10, 10.1], [40, 40.2], [270, 270.9], [320, 320.8], [500, 501]]
        # 结果:
        # 样本0: [10+270, 10.1+270.9] = [280, 281]  (专家0权重0.1 + 专家1权重0.9)
        # 样本1: [40+320, 40.2+320.8] = [360, 361]  (专家1权重0.2 + 专家0权重0.8) 
        # 样本2: [500, 501]                       (只有专家1参与，权重1.0)
        
        return combined

    def expert_to_gates(self):
        """
        返回每个专家接收到的门控权重列表，主要用于某些后续处理或调试。
        """
        result = torch.split(self._nonzero_gates, self._part_sizes, dim=0)
        print(f"\n各专家的门控权重:")
        for i, weights in enumerate(result):
            print(f"  专家{i}: {weights.flatten()}")
        # 专家0: tensor([0.1, 0.8])  # [样本0专家0权重, 样本1专家0权重]
        # 专家1: tensor([0.9, 0.2, 1.0])  # [样本0专家1权重, 样本1专家1权重, 样本2专家1权重]
        return result


# 专家网络示例 (简单的 MLP)
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):
    """
    稀疏门控混合专家层 (Sparsely-Gated Mixture-of-Experts Layer)。
    使用 1层前馈网络作为专家。
    
    参数:
    input_size: int - 输入维度
    output_size: int - 输出维度
    num_experts: int - 专家的数量
    hidden_size: int - 专家网络的隐藏层维度
    noisy_gating: bool - 是否使用噪声门控（训练时推荐 True）
    k: int - 每个样本选择 Top-k 个专家
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # 实例化专家网络列表 (这里简单的用 MLP 代替)
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        
        # 门控网络的参数: W_gate 用于计算原始分数，W_noise 用于计算噪声幅度
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax  = nn.Softmax(1)
        
        # 注册不需要梯度更新的 buffer，用于正态分布采样
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """
        计算变异系数的平方 (CV^2)。
        用于辅助损失函数，鼓励分布更加均匀。
        CV^2 = Var / Mean^2
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """
        当不使用噪声Top-k时，计算每个专家的负载。
        负载 = 对应门控权重大于0的样本数。
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        NoisyTopKGating 的辅助函数。
        计算值进入 Top-k 的概率，即 P(logit > Threshold)。
        这是 Load Loss 可导的关键，通过正态分布的 CDF 计算。
        
        参数:
        clean_values: 原始 logit (无噪声), [batch, n] 即 [batch_size, num_experts]
        noisy_values: 加噪后的 logits, [batch, n] 即 [batch_size, num_experts]
        noise_stddev: 噪声标准差 sigma, [batch, n] 即 [batch_size, num_experts]
        noisy_top_values: Top-(k+1) 的加噪 logits 值, [batch, k+1] (因为topk选择了k+1个用于计算阈值)
        """
        batch = clean_values.size(0)  # batch_size
        m = noisy_top_values.size(1)  # k+1
        top_values_flat = noisy_top_values.flatten()  # [batch * (k+1)]

        # 计算阈值：选取第 k 大的值作为基准
        # 注意：这里逻辑稍微复杂，因为它是根据 noisy_values 排序的
        # threshold_positions_if_in: [batch]，每个batch的第k个位置 (0-based: k)
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        # threshold_if_in: [batch, 1]，每个batch的第k+1大的值
        
        ###### 保级阈值
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        
        # is_in: [batch, num_experts]，判断每个logit是否超过阈值
        is_in = torch.gt(noisy_values, threshold_if_in)
        
        # 稍微调整阈值索引位置 (第k-1大的值，作为out的阈值)
        threshold_positions_if_out = threshold_positions_if_in - 1  # [batch]
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)  # [batch, 1]
        
        # 使用正态分布 CDF 计算概率
        normal = Normal(self.mean, self.std)
        # prob_if_in: [batch, num_experts]，假设在Top-k内，计算超过threshold_if_in的概率
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        # prob_if_out: [batch, num_experts]，假设不在Top-k内，计算超过threshold_if_out的概率
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        
        # prob: [batch, num_experts]，根据is_in选择对应的概率
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        带噪声的 Top-k 门控机制 (Noisy Top-k Gating).
        见论文 section 2.
        
        参数:
          x: 输入张量 [batch_size, input_size]
        
        返回:
          gates: 门控权重 [batch_size, num_experts]
          load: 负载均衡辅助张量 [num_experts]
        """
        # 1. 计算原始分数 h = x @ W_gate
        # x: [batch_size, input_size], w_gate: [input_size, num_experts] -> clean_logits: [batch_size, num_experts]
        clean_logits = x @ self.w_gate
        
        if self.noisy_gating and train:
            # 2. 计算噪声标准差 sigma = Softplus(x @ W_noise)
            # x: [batch_size, input_size], w_noise: [input_size, num_experts] -> raw_noise_stddev: [batch_size, num_experts]
            raw_noise_stddev = x @ self.w_noise
            # softplus后 + epsilon -> noise_stddev: [batch_size, num_experts]
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            
            # 3. 加入噪声 logits = h + noise
            # torch.randn_like(clean_logits): 生成与clean_logits同形状的随机噪声 [batch_size, num_experts]
            # noisy_logits: [batch_size, num_experts]
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 4. 计算 Top-k
        # logits: [batch_size, num_experts] -> softmax后仍 [batch_size, num_experts]
        logits = self.softmax(logits)
        # topk: 返回top_logits [batch_size, k], top_indices [batch_size, k]
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1) # top_logits: [batch_size, k+1]多选择了一个
        top_k_logits = top_logits[:, :self.k]  # [batch_size, k]
        top_k_indices = top_indices[:, :self.k]  # [batch_size, k]
        
        # 5. 归一化被选中的权重
        # top_k_gates: [batch_size, k] 归一化后仍 [batch_size, k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        # 构建稀疏的 gates 张量 (未选中置0)
        # zeros: [batch_size, num_experts], scatter后 gates: [batch_size, num_experts]
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 6. 计算 Load (用于辅助损失)
        if self.noisy_gating and self.k < self.num_experts and train:
            # 训练且有噪声时，使用概率估计 Load
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)  # [num_experts]
        else:
            # 否则直接统计硬选择次数
            load = self._gates_to_load(gates)  # [num_experts]
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """
        前向传播
        
        参数:
          x: 输入张量 [batch_size, input_size]
        
        返回:
          y: 混合后的输出 [batch_size, output_size]
          loss: 负载均衡辅助损失 (标量)
        """
        # 1. 门控选择
        # x: [batch_size, input_size] -> gates: [batch_size, num_experts], load: [num_experts]
        gates, load = self.noisy_top_k_gating(x, self.training)
        
        # 2. 计算辅助损失
        # importance: 门控权重之和 [num_experts]
        importance = gates.sum(0)
        # loss: 标量 (Importance Loss + Load Loss)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # 3. 分发 -> 专家计算 -> 组合
        dispatcher = SparseDispatcher(self.num_experts, gates)
        # expert_inputs: 列表，每个元素 [expert_batch_size_i, input_size] (分发后的输入)
        expert_inputs = dispatcher.dispatch(x)
        # gates: 列表，每个元素 [expert_batch_size_i, 1] (对应权重)
        gates = dispatcher.expert_to_gates()
        # expert_outputs: 列表，每个元素 [expert_batch_size_i, output_size] (专家输出)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        # y: [batch_size, output_size] (加权组合后的输出)
        y = dispatcher.combine(expert_outputs)
        
        return y, loss