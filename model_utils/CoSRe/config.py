from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class COSREConfig:
    """
    Hyper-parameters for COSRE routing & pruning.
    """
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None

    # target retention ratios
    rho_text_target: float = 0.7
    rho_vis_target: float = 0.4

    # temperature for soft routing (if you add it later)
    tau: float = 0.5

    # loss weights
    lambda_ratio: float = 1.0
    lambda_soft: float = 1.0
    lambda_hard: float = 1.0

    # always-keep prefix length
    prefix_kappa: int = 16
