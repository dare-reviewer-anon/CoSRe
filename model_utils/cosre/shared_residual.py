
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class SharedResidualConfig:
    hidden_size: int
    num_slots: int = 32
    num_heads: int = 8
    dropout: float = 0.0

    # Projection control
    use_modality_projections: bool = True  # WT/WV in the paper-style design

    fusion_mode: str = "scalar"


    residual_in_projected_space: bool = True


def _make_linear(in_dim: int, out_dim: int, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class SharedResidualFactorizer(nn.Module):

    def __init__(self, cfg: SharedResidualConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        h = cfg.num_heads
        if d % h != 0:
            raise ValueError(f"hidden_size={d} must be divisible by num_heads={h}.")

        # Optional modality projections
        if cfg.use_modality_projections:
            self.WT = _make_linear(d, d, bias=False)
            self.WV = _make_linear(d, d, bias=False)
        else:
            self.WT = None
            self.WV = None

        # Learnable slots: (K, d)
        self.slots = nn.Parameter(torch.randn(cfg.num_slots, d) * 0.02)

        # MultiheadAttention modules (batch_first=True for (B, L, d))
        self.attn_slots_T = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.attn_slots_V = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)

        self.attn_explain_T = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.attn_explain_V = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)

        # LayerNorm on shared slots
        self.ln_S = nn.LayerNorm(d)

        # Fusion gate alpha
        fm = cfg.fusion_mode.lower().strip()
        if fm not in ("scalar", "per_slot", "per_dim"):
            raise ValueError(f"fusion_mode must be one of ['scalar','per_slot','per_dim'], got {cfg.fusion_mode}.")
        self.fusion_mode = fm

        if fm == "scalar":
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fm == "per_slot":
            self.alpha = nn.Parameter(torch.full((cfg.num_slots,), 0.5))
        else:  # per_dim
            self.alpha = nn.Parameter(torch.full((d,), 0.5))

        # Optional inverse projections if you ever want residuals in original space
        if (not cfg.residual_in_projected_space) and cfg.use_modality_projections:
            # Use tied "pseudo-inverse" via another linear; you may also tie weights if desired.
            self.inv_WT = _make_linear(d, d, bias=False)
            self.inv_WV = _make_linear(d, d, bias=False)
        else:
            self.inv_WT = None
            self.inv_WV = None

    @staticmethod
    def _to_key_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if mask.dtype == torch.bool:
            #   - Else it's likely a pad-mask.
            total = mask.numel()
            keep_count = int(mask.sum().item())
            if keep_count > total // 2:
                return ~mask  # keep-mask -> pad-mask
            return mask       # already pad-mask
        else:
            # numeric: treat >0 as keep
            keep = mask > 0
            return ~keep

    def _project(self, T: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.WT is not None:
            T = self.WT(T)
        if self.WV is not None:
            V = self.WV(V)
        return T, V

    def _fusion_alpha(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Return alpha broadcastable to (B, K, d).
        """
        a = torch.sigmoid(self.alpha).to(device=device, dtype=dtype)
        if self.fusion_mode == "scalar":
            # (1,1,1)
            return a.view(1, 1, 1)
        if self.fusion_mode == "per_slot":
            # (1,K,1)
            return a.view(1, -1, 1)
        # per_dim: (1,1,d)
        return a.view(1, 1, -1)

    def forward(
        self,
        text_tokens: torch.Tensor,
        vis_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vis_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
          text_tokens: (B, Nt, d)
          vis_tokens:  (B, Nv, d)
          text_mask:   optional (B, Nt) keep-mask or pad-mask (see _to_key_padding_mask)
          vis_mask:    optional (B, Nv) keep-mask or pad-mask
          return_attn: whether to return attention weights in aux

        Returns:
          S:   (B, K, d) shared semantics slots
          RT:  (B, Nt, d) text residuals
          RV:  (B, Nv, d) visual residuals
          aux: dict with optional attention maps and intermediates
        """
        if text_tokens.dim() != 3 or vis_tokens.dim() != 3:
            raise ValueError("text_tokens and vis_tokens must be 3D tensors: (B, L, d).")
        B, Nt, d = text_tokens.shape
        B2, Nv, d2 = vis_tokens.shape
        if B2 != B or d2 != d:
            raise ValueError("Batch size and hidden dim must match between text and vision tokens.")
        if d != self.cfg.hidden_size:
            raise ValueError(f"hidden dim mismatch: got {d}, expected {self.cfg.hidden_size}")

        # Project modalities into shared space
        Tt, Vt = self._project(text_tokens, vis_tokens)  # (B,Nt,d), (B,Nv,d)

        # Prepare slots for batch
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)  # (B,K,d)

        # Convert masks to key_padding_mask (True means ignore)
        t_kpm = self._to_key_padding_mask(text_mask)  # (B,Nt) or None
        v_kpm = self._to_key_padding_mask(vis_mask)   # (B,Nv) or None

        # Slots attend to each modality
        # query=slots, key/value=modality tokens
        ST, attn_ST = self.attn_slots_T(query=slots, key=Tt, value=Tt, key_padding_mask=t_kpm, need_weights=return_attn)
        SV, attn_SV = self.attn_slots_V(query=slots, key=Vt, value=Vt, key_padding_mask=v_kpm, need_weights=return_attn)

        # Fuse into shared slots S
        alpha = self._fusion_alpha(B, device=Tt.device, dtype=Tt.dtype)  # broadcastable to (B,K,d)
        S = self.ln_S(alpha * ST + (1.0 - alpha) * SV)  # (B,K,d)

        # Explain each modality from shared slots
        # query=modality tokens, key/value=shared slots
        That, attn_That = self.attn_explain_T(query=Tt, key=S, value=S, need_weights=return_attn)
        Vhat, attn_Vhat = self.attn_explain_V(query=Vt, key=S, value=S, need_weights=return_attn)

        # Residuals
        RT_proj = Tt - That
        RV_proj = Vt - Vhat

        if self.cfg.residual_in_projected_space or (self.inv_WT is None):
            RT = RT_proj
            RV = RV_proj
        else:
            # map residuals back to original token space
            RT = self.inv_WT(RT_proj)
            RV = self.inv_WV(RV_proj)

        aux: Dict[str, torch.Tensor] = {
            "T_proj": Tt,
            "V_proj": Vt,
            "That": That,
            "Vhat": Vhat,
            "alpha": alpha.squeeze().detach() if alpha.numel() > 1 else alpha.detach(),
        }
        if return_attn:
            # shapes from MultiheadAttention (batch_first=True):
            # attn weights: (B, tgt_len, src_len) if average_attn_weights=True (default)
            aux.update(
                {
                    "attn_ST": attn_ST,       # slots -> text
                    "attn_SV": attn_SV,       # slots -> vision
                    "attn_That": attn_That,   # text -> slots
                    "attn_Vhat": attn_Vhat,   # vision -> slots
                }
            )

        return S, RT, RV, aux

