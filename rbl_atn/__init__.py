"""RBL-ATN (Rule-Bounded Attention) for parameter-golf / train_gpt."""

from .attention import RBLMultiHeadAttention
from .gpt_attention import RBLCausalSelfAttention

__all__ = ["RBLCausalSelfAttention", "RBLMultiHeadAttention"]
