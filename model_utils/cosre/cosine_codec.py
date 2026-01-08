from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import math
import torch
import torch.nn as nn


@dataclass
class CosineCodecConfig:
    # Visual grid size (for reshape convenience; can also be inferred)
    block_size: int = 8                  # b
    keep_hw: Tuple[int, int] = (4, 4)    # (kh, kw) per block after reconstruction
    base_delta: float = 0.5              # quantization base step
    delta_mode: str = "linear"           # "linear" or "quadratic"
    hard_round: bool = True              # True: torch.round; False: STE-style round
    clamp_q: Optional[float] = None      # optional clamp magnitude of quantized coeffs


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def build_ortho_dct_matrix(N: int, device=None, dtype=None) -> torch.Tensor:
    
    n = torch.arange(N, device=device, dtype=dtype).view(1, -1)  # (1, N)
    k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)  # (N, 1)
    mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))         # (N, N)
    mat[0, :] *= math.sqrt(1.0 / N)
    mat[1:, :] *= math.sqrt(2.0 / N)
    return mat


def build_delta_table(b: int, base_delta: float, mode: str, device=None, dtype=None) -> torch.Tensor:
    
    u = torch.arange(b, device=device, dtype=dtype).view(-1, 1)
    v = torch.arange(b, device=device, dtype=dtype).view(1, -1)
    mode = mode.lower().strip()
    if mode == "linear":
        return base_delta * (1.0 + u + v)
    if mode == "quadratic":
        return base_delta * (1.0 + u * u + v * v)
    raise ValueError(f"Unknown delta_mode={mode}. Use 'linear' or 'quadratic'.")


def build_omega_mask(b: int, keep_hw: Tuple[int, int], device=None, dtype=None) -> torch.Tensor:
    
    kh, kw = keep_hw
    if not (1 <= kh <= b and 1 <= kw <= b):
        raise ValueError(f"keep_hw must be within [1,b], got {keep_hw} for b={b}.")
    omega = torch.zeros((b, b), device=device, dtype=dtype)
    omega[:kh, :kw] = 1.0
    return omega


class BlockwiseCosineCodec(nn.Module):
    

    def __init__(self, cfg: CosineCodecConfig):
        super().__init__()
        self.cfg = cfg
        b = cfg.block_size
        kh, kw = cfg.keep_hw
        if kh > b or kw > b:
            raise ValueError(f"keep_hw={cfg.keep_hw} must be <= block_size={b}.")

        # Buffers will be built lazily on first forward (to match device/dtype).
        self.register_buffer("_D", torch.empty(0), persistent=False)
        self.register_buffer("_Delta", torch.empty(0), persistent=False)
        self.register_buffer("_Omega", torch.empty(0), persistent=False)

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype):
        b = self.cfg.block_size
        if self._D.numel() == 0 or self._D.device != device or self._D.dtype != dtype:
            D = build_ortho_dct_matrix(b, device=device, dtype=dtype)
            Delta = build_delta_table(b, self.cfg.base_delta, self.cfg.delta_mode, device=device, dtype=dtype)
            Omega = build_omega_mask(b, self.cfg.keep_hw, device=device, dtype=dtype)
            self._D = D
            self._Delta = Delta
            self._Omega = Omega

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D DCT per channel via matrix multiplies.
        x: (Bn, b, b, D)
        return: (Bn, b, b, D)
        """
        D = self._D
        # Move channels to do (b,b) matmul with broadcasting:
        # (Bn, b, b, D) -> (Bn, D, b, b)
        y = x.permute(0, 3, 1, 2)
        y = torch.matmul(D, y)       # (b,b) @ (Bn,D,b,b) -> broadcast
        y = torch.matmul(y, D.t())
        return y.permute(0, 2, 3, 1)

    def _idct2(self, c: torch.Tensor) -> torch.Tensor:
        """
        Inverse 2D DCT per channel using orthonormal basis.
        c: (Bn, b, b, D)
        return: (Bn, b, b, D)
        """
        D = self._D
        y = c.permute(0, 3, 1, 2)
        y = torch.matmul(D.t(), y)
        y = torch.matmul(y, D)
        return y.permute(0, 2, 3, 1)

    def _quantize(self, c: torch.Tensor) -> torch.Tensor:
        """
        Quantize coefficients with Δ[u,v].
        c: (Bn, b, b, D)
        """
        Delta = self._Delta[..., None]  # (b,b,1)
        z = c / Delta
        if self.cfg.hard_round:
            q = torch.round(z)
        else:
            q = _RoundSTE.apply(z)
        if self.cfg.clamp_q is not None:
            q = torch.clamp(q, -self.cfg.clamp_q, self.cfg.clamp_q)
        return q

    def _dequantize(self, q: torch.Tensor) -> torch.Tensor:
        Delta = self._Delta[..., None]
        return q * Delta

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None,
        flatten: bool = False,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if x.dim() == 4:
            B, H, W, Dch = x.shape
        elif x.dim() == 3:
            if grid_hw is None:
                raise ValueError("grid_hw=(H,W) must be provided when input is (B,N,D).")
            B, N, Dch = x.shape
            H, W = grid_hw
            if N != H * W:
                raise ValueError(f"Token count N={N} must equal H*W={H*W}.")
            x = x.view(B, H, W, Dch)
        else:
            raise ValueError("Input x must be (B,H,W,D) or (B,N,D).")

        b = self.cfg.block_size
        kh, kw = self.cfg.keep_hw
        if H % b != 0 or W % b != 0:
            raise ValueError(f"H,W must be divisible by block_size={b}, got H={H}, W={W}.")

        self._ensure_buffers(device=x.device, dtype=x.dtype)

        # -------- blockify --------
        # (B,H,W,D) -> (B, Hb, Wb, b, b, D) -> (Bn, b, b, D)
        Hb, Wb = H // b, W // b
        blocks = (
            x.view(B, Hb, b, Wb, b, Dch)
             .permute(0, 1, 3, 2, 4, 5)
             .contiguous()
             .view(B * Hb * Wb, b, b, Dch)
        )

        c = self._dct2(blocks)

        q = self._quantize(c)  # integer-like (float tensor)

        c_hat = self._dequantize(q)
        rec = self._idct2(c_hat)  # (Bn,b,b,D)

        
        rec_small = rec[:, :kh, :kw, :]  # (Bn, kh, kw, D)

        # -------- Unblock --------
        H2, W2 = Hb * kh, Wb * kw
        y = (
            rec_small.view(B, Hb, Wb, kh, kw, Dch)
                     .permute(0, 1, 3, 2, 4, 5)
                     .contiguous()
                     .view(B, H2, W2, Dch)
        )

        if flatten:
            y_out = y.view(B, H2 * W2, Dch)
        else:
            y_out = y

        aux: Dict[str, torch.Tensor] = {}
        if return_aux:
            # sparsity proxy: fraction of zeroed coefficients after Ω+quant
            with torch.no_grad():
                # q has Ω masked already
                zeros = (q == 0).float().mean()
                aux["q_zero_frac"] = zeros.detach()
                aux["H_out"] = torch.tensor(H2, device=x.device)
                aux["W_out"] = torch.tensor(W2, device=x.device)
                aux["block_size"] = torch.tensor(b, device=x.device)
                aux["keep_h"] = torch.tensor(kh, device=x.device)
                aux["keep_w"] = torch.tensor(kw, device=x.device)

        return y_out, aux


