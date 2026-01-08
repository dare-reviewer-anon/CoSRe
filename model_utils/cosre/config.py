from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CoSReConfig:
   
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None


    image_hw: Tuple[int, int] = (32, 32)

    # Blockwise DCT parameters
    block_size: int = 8                 # block size b×b
    keep_h: int = 4                     # low-frequency rows kept per block
    keep_w: int = 4                     # low-frequency cols kept per block

    # Quantization
    base_delta: float = 0.5             # base quantization step
    delta_mode: str = "linear"          # "linear" | "quadratic"
    hard_round: bool = True             # True: round(); False: STE-style
    clamp_q: Optional[float] = None     # optional clamp on quantized coeffs


    num_slots: int = 32                 # number of shared semantic slots K
    fusion_mode: str = "scalar"         # "scalar" | "per_slot" | "per_dim"
    dropout: float = 0.0


    prefix_kappa: int = 0

    enable_cosre: bool = False
