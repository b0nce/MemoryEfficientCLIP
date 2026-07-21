"""Memory-efficient CLIP, LiT, and Qwen3 losses (Triton kernels).

`pip install memeff`, then `from memeff import MemoryEfficientCLIPLoss` -- every
public class is re-exported here.
"""
__version__ = "0.2.0"

from .clip_loss import MemoryEfficientCLIPLoss, StableMemoryEfficientCLIPLoss
from .lit_loss import MemoryEfficientLiTLoss, StableMemoryEfficientLiTLoss
from .distributed_clip_loss import DistributedMemoryEfficientCLIPLoss
from .distributed_lit_loss import DistributedMemoryEfficientLiTLoss
from .clip_qwen3_loss import (
    MemoryEfficientQwen3Loss, DistributedMemoryEfficientQwen3Loss)
from .lit_qwen3_loss import (
    MemoryEfficientLiTQwen3Loss, DistributedMemoryEfficientLiTQwen3Loss)
from .matryoshka import MatryoshkaLoss
from .mrl_qwen3_loss import (
    MemoryEfficientMatryoshkaQwen3Loss, MemoryEfficientMatryoshkaLiTQwen3Loss)

__all__ = [
    "MemoryEfficientCLIPLoss",
    "StableMemoryEfficientCLIPLoss",
    "MemoryEfficientLiTLoss",
    "StableMemoryEfficientLiTLoss",
    "DistributedMemoryEfficientCLIPLoss",
    "DistributedMemoryEfficientLiTLoss",
    "MemoryEfficientQwen3Loss",
    "DistributedMemoryEfficientQwen3Loss",
    "MemoryEfficientLiTQwen3Loss",
    "DistributedMemoryEfficientLiTQwen3Loss",
    "MatryoshkaLoss",
    "MemoryEfficientMatryoshkaQwen3Loss",
    "MemoryEfficientMatryoshkaLiTQwen3Loss",
]
