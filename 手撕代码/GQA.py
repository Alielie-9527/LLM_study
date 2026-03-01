"""
Unified Attention supporting MHA/MQA/GQA
Compatible with LLaMA-2/3, Mixtral architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class UnifiedAttention(nn.Module):
    """
    工业级统一Attention实现
    支持: MHA (num_kv_heads=num_heads), GQA (num_kv_heads<num_heads), MQA (num_kv_heads=1)
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,  # None表示MHA
        head_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,  # RoPE基频
        rope_scaling: Optional[dict] = None,  # 长文本外推配置
        attn_dropout: float = 0.0,
        use_flash_attn: bool = True,
        bias: bool = False,  # LLaMA风格无偏置
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_groups = self.num_heads // self.num_kv_heads  # GQA分组数
        
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads({num_heads}) must be divisible by num_kv_heads({num_kv_heads})"
        
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.scaling = self.head_dim ** -0.5
        
        # 投影层
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # RoPE旋转位置编码
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_theta,
            scaling=rope_scaling,
        )
        
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        
        # KV Cache管理
        self.register_buffer(
            "k_cache",
            torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "v_cache", 
            torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim),
            persistent=False,
        )
        self.cache_seq_len = 0
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码（用于padding或causal mask）
            position_ids: 位置编码ID
            past_key_value: 历史KV Cache (k, v)
            use_cache: 是否使用并更新KV Cache
            output_attentions: 是否返回注意力权重（训练时调试用）
        
        Returns:
            attn_output: 注意力输出 [batch, seq_len, hidden_dim]
            present_key_value: 更新后的KV Cache（如果use_cache=True）
            attn_weights: 注意力权重（如果output_attentions=True）
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 线性投影
        # Q: [batch, seq_len, num_heads * head_dim]
        # K,V: [batch, seq_len, num_kv_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 2.  reshape为多头格式 [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用RoPE位置编码
        cos, sin = self.rope(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # 4. 处理KV Cache（推理时）
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)
        
        past_key_value_out = (key_states, value_states) if use_cache else None
        
        # 5. GQA: 扩展KV到与Q相同的头数（通过重复）
        # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        if self.num_groups > 1:
            key_states = repeat_kv(key_states, self.num_groups)
            value_states = repeat_kv(value_states, self.num_groups)
        
        # 6. 注意力计算
        attn_weights = None
        if self.use_flash_attn and not output_attentions:
            # Flash Attention路径（最快）
            attn_output = self._flash_attention(query_states, key_states, value_states, attention_mask)
        else:
            # 标准路径（支持返回attention weights）
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask, output_attentions
            )
        
        # 7. reshape并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.attn_dropout(attn_output)
        
        return attn_output, past_key_value_out, attn_weights
    
    def _flash_attention(
        self,
        query: torch.Tensor,  # [batch, num_heads, q_len, head_dim]
        key: torch.Tensor,    # [batch, num_heads, kv_len, head_dim] (已扩展)
        value: torch.Tensor,  # [batch, num_heads, kv_len, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用PyTorch原生Flash Attention (scaled_dot_product_attention)"""
        # 现代写法：使用 torch.nn.attention.sdp_kernel 控制后端选择
        from torch.nn.attention import SDPBackend, sdp_kernel
        
        # 排除掉慢速的 MATH 后端，仅允许 Flash 或 Mem-Efficient
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        
        with sdp_kernel(backends=backends):
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=attention_mask is None,  # 无mask时使用causal mask
            )
        return attn_output
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """标准注意力计算（用于调试或需要attention weights时）"""
        
        # 计算注意力分数: [batch, num_heads, q_len, kv_len]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        # 应用mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, value)
        
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA核心操作: 重复KV头以匹配Q头数
    输入: [batch, num_kv_heads, seq_len, head_dim]
    输出: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    
    内存高效实现: 扩展维度而非实际复制数据
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # 方法1: 实际扩展（内存占用高但兼容性好）
    # return hidden_states.repeat_interleave(n_rep, dim=1)
    
    # 方法2: 广播扩展（内存高效，利用stride tricks）
    # [batch, num_kv_heads, 1, seq_len, head_dim] -> [batch, num_kv_heads, n_rep, seq_len, head_dim]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 实现
    支持动态NTK-aware缩放和长文本外推
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling: Optional[dict] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 长文本缩放（如YaRN, NTK-aware）
        self.scaling = scaling
        if scaling is not None:
            self._apply_scaling(scaling)
        
        self._set_cos_sin_cache(seq_len=max_seq_len)
    
    def _apply_scaling(self, scaling: dict):
        """应用RoPE缩放策略"""
        scale_type = scaling.get("type", "linear")
        factor = scaling.get("factor", 1.0)
        
        if scale_type == "linear":
            # 线性插值
            self.max_seq_len = int(self.max_seq_len * factor)
        elif scale_type == "ntk":
            # NTK-aware缩放
            self.base = self.base * factor ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        elif scale_type == "yarn":
            # YaRN缩放（更复杂，需额外参数）
            pass  # 简化处理
    
    def _set_cos_sin_cache(self, seq_len: int):
        """预计算cos/sin缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 外积: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # 重复: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int):
        """返回cos, sin用于旋转"""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转半个维度: [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用RoPE到Q和K
    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    if position_ids is not None:
        # 动态索引（用于推理时position不连续）
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, head_dim]
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    
    # 旋转操作: q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==================== 工厂函数 ====================

def create_attention(
    attn_type: str = "mha",  # "mha", "gqa", "mqa"
    hidden_dim: int = 4096,
    num_heads: int = 32,
    **kwargs
) -> UnifiedAttention:
    """工厂函数：根据类型创建Attention"""
    config = {
        "mha": {"num_kv_heads": num_heads},
        "gqa": {"num_kv_heads": num_heads // 4},  # 如32->8
        "mqa": {"num_kv_heads": 1},
    }
    
    if attn_type not in config:
        raise ValueError(f"Unknown attn_type: {attn_type}")
    
    return UnifiedAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        **config[attn_type],
        **kwargs
    )


# ==================== 测试与验证 ====================

def test_attention():
    """验证MHA/GQA/MQA的正确性和内存差异"""
    batch_size, seq_len, hidden_dim = 2, 1024, 4096
    num_heads = 32
    
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    for attn_type in ["mha", "gqa", "mqa"]:
        print(f"\n{'='*50}")
        print(f"Testing {attn_type.upper()}")
        print(f"{'='*50}")
        
        attn = create_attention(
            attn_type=attn_type,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        ).cuda()
        
        # 统计参数量
        total_params = sum(p.numel() for p in attn.parameters())
        trainable_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"KV projection params: {attn.k_proj.weight.numel() + attn.v_proj.weight.numel():,}")
        print(f"Num KV heads: {attn.num_kv_heads}")
        print(f"Compression ratio: {num_heads}/{attn.num_kv_heads} = {num_heads//attn.num_kv_heads}x")
        
        # 前向测试
        with torch.amp.autocast(device_type="cuda"):  # 现代混合精度写法
            out, past_kv, _ = attn(x, use_cache=True)
        
        print(f"Output shape: {out.shape}")
        print(f"KV Cache shape: {past_kv[0].shape}")
        print(f"KV Cache memory: {past_kv[0].numel() * 2 * 2 / 1024**2:.2f} MB (FP16)")  # K+V
        
        # 推理测试（增量解码）
        past_k, past_v = past_kv
        next_token = torch.randn(batch_size, 1, hidden_dim).cuda()
        out_next, past_kv_next, _ = attn(
            next_token, 
            past_key_value=(past_k, past_v),
            use_cache=True
        )
        print(f"Incremental decode output: {out_next.shape}")
        
        # 验证梯度
        loss = out.sum()
        loss.backward()
        print(f"Gradients computed successfully")


if __name__ == "__main__":
    test_attention()