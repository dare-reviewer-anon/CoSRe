from __future__ import annotations

from typing import Optional, Dict, List

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import COSREConfig
from .diff_topk import DifferentiableTopK
from .attention import EfficientAttention  


logger = logging.getLogger(__name__)



class COSREController(nn.Module):
    def __init__(self, cfg: COSREConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("num_routed_steps", torch.zeros(1, dtype=torch.long), persistent=False)
        self._last_losses: Dict[str, torch.Tensor] = {}
        self.diff_topk = DifferentiableTopK(temperature=cfg.tau)


    @property
    def last_losses(self) -> Dict[str, torch.Tensor]:
        return self._last_losses

    def route(
        self,
        importance: torch.Tensor,          # [B, S]
        modality_mask: Optional[torch.Tensor],
        *,
        is_visual: bool,
        layer_idx: int,
        training: bool,
    ):
        if importance.dim() == 3 and importance.size(-1) == 1:
            importance = importance.squeeze(-1)

        B, S = importance.shape
        cfg = self.cfg

        if modality_mask is None:
            modality_mask = torch.ones_like(importance, dtype=torch.bool)
        else:
            modality_mask = modality_mask.bool()

        rho_target = cfg.rho_vis_target if is_visual else cfg.rho_text_target
        target_tokens = (modality_mask.sum(dim=1).float() * rho_target).clamp(min=1.0)

        prefix_len = min(cfg.prefix_kappa, S)
        prefix_mask = torch.zeros_like(modality_mask)
        prefix_mask[:, :prefix_len] = True

        effective_mask = modality_mask.clone()
        effective_mask[:, :prefix_len] = False

        keep_mask = prefix_mask.clone()

        for b in range(B):
            valid_positions = torch.nonzero(effective_mask[b], as_tuple=False).squeeze(-1)
            if valid_positions.numel() == 0:
                continue

            k = int(target_tokens[b].item())
            k = min(k, valid_positions.numel())

            scores = importance[b, valid_positions].unsqueeze(0)  
            soft_mask = self.diff_topk(scores, k, training=training)[0]  
            hard_mask = (soft_mask > 0.5)
            selected_positions = valid_positions[hard_mask]
            keep_mask[b, selected_positions] = True


        loss_ratio = importance.new_zeros(())
        loss_soft = importance.new_zeros(())
        loss_hard = importance.new_zeros(())

        if training:
            kept_counts = (keep_mask & modality_mask).sum(dim=1).float()
            actual_ratio = kept_counts / (modality_mask.sum(dim=1).float() + 1e-6)
            ratio_target = torch.full_like(actual_ratio, rho_target)

            loss_ratio = F.l1_loss(actual_ratio, ratio_target)

            if is_visual:
                dropped = modality_mask & (~keep_mask)
                if dropped.any():
                    loss_soft = importance[dropped].mean()

                margin = rho_target - actual_ratio
                loss_hard = torch.clamp(margin, min=0).mean()

        total_ratio = cfg.lambda_ratio * loss_ratio
        total_soft = cfg.lambda_soft * loss_soft
        total_hard = cfg.lambda_hard * loss_hard

        loss_dict = {
            "CoSRe_ratio_loss": total_ratio,
            "CoSRe_soft_loss": total_soft,
            "CoSRe_hard_loss": total_hard,
        }

        self._last_losses = loss_dict
        self.num_routed_steps += 1

        return keep_mask, loss_dict


class ModalityRouter(nn.Module):
    """
    Thin wrapper around COSREController, so you can easily swap routing logic later.
    """

    def __init__(self, controller: COSREController):
        super().__init__()
        self.controller = controller

    def forward(
        self,
        importance: torch.Tensor,
        modality_mask: Optional[torch.Tensor],
        *,
        is_visual: bool,
        layer_idx: int,
        training: bool,
    ):
        return self.controller.route(
            importance=importance,
            modality_mask=modality_mask,
            is_visual=is_visual,
            layer_idx=layer_idx,
            training=training,
        )


def gather_prune_kv(
    key: torch.Tensor,     # [B, H, S, D]
    value: torch.Tensor,   # [B, H, S, D]
    keep_mask: torch.Tensor,   # [B, S]
):
    assert key.dim() == 4 and value.dim() == 4
    B, H, S, D = key.shape

    keep_mask = keep_mask.bool()
    kept_per_b = keep_mask.sum(dim=1)

    max_kept = int(kept_per_b.max().item())
    min_kept = int(kept_per_b.min().item())

    first_mask = keep_mask[0]
    base_idx = torch.nonzero(first_mask, as_tuple=False).squeeze(-1)

    if base_idx.numel() < max_kept:
        extra = []
        for s in range(S):
            if s not in base_idx and len(extra) < (max_kept - base_idx.numel()):
                extra.append(s)
        base_idx = torch.cat([base_idx, torch.tensor(extra, device=base_idx.device)], dim=0)

    base_idx = base_idx[:max_kept]

    key_new = key.index_select(dim=2, index=base_idx)
    value_new = value.index_select(dim=2, index=base_idx)

    return key_new.contiguous(), value_new.contiguous()



def _find_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    cands: List[nn.Module] = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = getattr(model.model, "layers")
        if isinstance(layers, (nn.ModuleList, list)):
            cands = list(layers)

    if not cands and hasattr(model, "model") and hasattr(model.model, "decoder"):
        dec = model.model.decoder
        if hasattr(dec, "layers") and isinstance(dec.layers, (nn.ModuleList, list)):
            cands = list(dec.layers)

    if not cands and hasattr(model, "transformer"):
        tr = model.transformer
        for name in ["layers", "blocks"]:
            if hasattr(tr, name):
                blk = getattr(tr, name)
                if isinstance(blk, (nn.ModuleList, list)):
                    cands = list(blk)
                    break

    if not cands and hasattr(model, "layers"):
        layers = getattr(model, "layers")
        if isinstance(layers, (nn.ModuleList, list)):
            cands = list(layers)

    if not cands:
        for m in model.modules():
            cls = m.__class__.__name__.lower()
            if any(k in cls for k in ["block", "decoderlayer", "transformerlayer"]):
                cands.append(m)
        cands = list(dict.fromkeys(cands))

    return cands


def attach_CoSRe_to_anole(model: nn.Module, controller: COSREController):
    blocks = _find_transformer_blocks(model)

    if not blocks:
        logger.warning("[COSRE] Could not find Transformer blocks. COSRE disabled.")
        return model

    num_attn = 0
    for idx, block in enumerate(blocks):
        attn = None
        if hasattr(block, "self_attn"):
            attn = getattr(block, "self_attn")
            attn_attr_name = "self_attn"
        elif hasattr(block, "attention"):
            attn = getattr(block, "attention")
            attn_attr_name = "attention"
        else:
            continue

        if isinstance(attn, nn.Module):
            if not isinstance(attn, EfficientAttention):
                wrapped = EfficientAttention(attn)
            else:
                wrapped = attn

            setattr(wrapped.base_attn, "CoSRe_controller", controller)
            setattr(wrapped.base_attn, "CoSRe_layer_idx", idx)

            setattr(block, attn_attr_name, wrapped)

            num_attn += 1

    if num_attn == 0:
        logger.warning("[COSRE] No attention modules found inside blocks.")
    else:
        logger.info(f"[COSRE] Attached controller to {num_attn} attention layers.")

    setattr(model, "CoSRe_controller", controller)
    setattr(model, "CoSRe_enabled", True)

    return model
