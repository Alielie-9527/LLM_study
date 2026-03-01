# 现代 LLM 解码器中 causal mask 与 padding mask 生成工具
import torch
from torch import Tensor
from typing import Optional


def apply_decoder_masks(
    scores: Tensor,
    seq_len: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """在 attention scores 上叠加 causal mask 与 padding mask."""
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=scores.device, dtype=scores.dtype),
        diagonal=1,
    )
    scores[:, :, :, -seq_len:] += causal_mask

    if attention_mask is not None:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        scores = scores + extended_attention_mask

    return scores


if __name__ == "__main__":
    batch_size, num_heads, seq_len = 2, 4, 8
    scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])

    masked = apply_decoder_masks(scores, seq_len, padding_mask)
    print(masked)
