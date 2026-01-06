from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


def prune_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    keep_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if key_states.dim() != 4 or value_states.dim() != 4:
        raise ValueError(
            f"Expected key/value to be 4D [B, H, S, D], got "
            f"{tuple(key_states.shape)} and {tuple(value_states.shape)}"
        )

    B, H, S, D = key_states.shape
    if value_states.shape != (B, H, S, D):
        raise ValueError(
            f"value_states must have same shape as key_states. "
            f"Got {tuple(value_states.shape)} vs {tuple(key_states.shape)}"
        )

    if keep_mask.dim() != 2 or keep_mask.shape != (B, S):
        raise ValueError(
            f"keep_mask must be [B, S], got {tuple(keep_mask.shape)}"
        )

    keep_mask = keep_mask.to(dtype=torch.bool)

    kept_per_b = keep_mask.sum(dim=1)  # [B]
    max_kept = int(kept_per_b.max().item())
    min_kept = int(kept_per_b.min().item())

    if max_kept == 0:
        logger.warning(
            "[COSRE] prune_kv_cache: all entries in keep_mask are 0; "
            "disabling pruning for this forward pass."
        )
        idx = torch.arange(S, device=key_states.device, dtype=torch.long)
        idx = idx.unsqueeze(0).expand(B, S)  # [B, S]
        return key_states, value_states, attention_mask, idx

    if min_kept == 0:
        logger.warning(
            "[COSRE] prune_kv_cache: some batch elements have 0 kept tokens. "
            "They will be forced to keep at least one position (prefix)."
        )

    batch_indices = []
    for b in range(B):
        valid = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(-1)

        if valid.numel() == 0:
            # Force-keep position 0
            valid = torch.tensor([0], device=key_states.device, dtype=torch.long)

        if valid.numel() > max_kept:
            valid = valid[:max_kept]

        if valid.numel() < max_kept:
            pad_count = max_kept - valid.numel()
            pad_idx = valid[-1].repeat(pad_count)
            valid = torch.cat([valid, pad_idx], dim=0)

        batch_indices.append(valid)

    gather_indices = torch.stack(batch_indices, dim=0)

    # gather along sequence dim
    idx_expanded = gather_indices[:, None, :, None].expand(-1, H, -1, D)
    key_pruned = torch.gather(key_states, dim=2, index=idx_expanded)
    value_pruned = torch.gather(value_states, dim=2, index=idx_expanded)

    attn_pruned = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            # [B, 1, Q, S] -> gather over last dim
            attn_pruned = torch.gather(
                attention_mask,
                dim=-1,
                index=gather_indices[:, None, None, :].expand(
                    -1, attention_mask.size(1), attention_mask.size(2), -1
                ),
            )
        elif attention_mask.dim() == 2 and attention_mask.shape == (B, S):
            attn_pruned = torch.gather(attention_mask, dim=1, index=gather_indices)
        else:
            logger.warning(
                "[COSRE] prune_kv_cache: attention_mask has unsupported shape %s; "
                "it will be returned unchanged.",
                tuple(attention_mask.shape),
            )
            attn_pruned = attention_mask

    return (
        key_pruned.contiguous(),
        value_pruned.contiguous(),
        attn_pruned,
        gather_indices,
    )


class EfficientAttention(nn.Module):

    def __init__(self, base_attn: nn.Module):
        super().__init__()
        self.base_attn = base_attn

    def forward(
        self,
        *args,
        keep_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            keep_mask: [B, S] or [B, 1, S] bool/int mask.
                       1 = keep KV position, 0 = prune.
        """
        # No pruning requested → vanilla behaviour
        if keep_mask is None:
            return self.base_attn(*args, **kwargs)

        captured = {}

        def capture_k(module, inp, out):
            captured["k"] = out

        def capture_v(module, inp, out):
            captured["v"] = out

        handles = []
        try:
            handles.append(self.base_attn.k_proj.register_forward_hook(capture_k))
            handles.append(self.base_attn.v_proj.register_forward_hook(capture_v))
        except AttributeError:
            # Fallback: if no k_proj / v_proj exist, just run without pruning
            logger.warning(
                "[COSRE] EfficientAttention: base_attn has no k_proj/v_proj; "
                "skipping KV capture."
            )
            return self.base_attn(*args, **kwargs)

        # --- run base attention ---
        out = self.base_attn(*args, **kwargs)

        # always clean up hooks
        for h in handles:
            h.remove()

        # safety: if we failed to capture K/V, just return
        if "k" not in captured or "v" not in captured:
            logger.warning(
                "[COSRE] EfficientAttention: K/V not captured; returning original output."
            )
            return out

        # Unpack output (expected: (attn_output, attn_weights, past_key_value))
        if not (isinstance(out, tuple) and len(out) == 3):
            # unknown output format, do not touch it
            return out

        attn_output, attn_weights, past = out

        # flatten keep_mask to [B, S]
        if keep_mask.dim() == 3 and keep_mask.size(1) == 1:
            keep_mask_flat = keep_mask[:, 0, :]
        elif keep_mask.dim() == 2:
            keep_mask_flat = keep_mask
        else:
            logger.warning(
                "[COSRE] EfficientAttention: keep_mask has unexpected shape %s; "
                "skipping pruning.",
                tuple(keep_mask.shape),
            )
            return out

        # captured k/v are [B, S, D_kv]; convert to [B, H_kv, S, d]
        key_states = captured["k"]
        value_states = captured["v"]

        if key_states.dim() != 3 or value_states.dim() != 3:
            logger.warning(
                "[COSRE] EfficientAttention: captured K/V have shape %s and %s; "
                "expected [B, S, D]. Skipping pruning.",
                tuple(key_states.shape),
                tuple(value_states.shape),
            )
            return out

        B, S, Dkv = key_states.shape
        H_kv = getattr(self.base_attn, "num_key_value_heads", None)
        if H_kv is None or Dkv % H_kv != 0:
            logger.warning(
                "[COSRE] EfficientAttention: cannot infer head_dim from "
                "Dkv=%d and num_key_value_heads=%s; skipping pruning.",
                Dkv,
                str(H_kv),
            )
            return out

        head_dim = Dkv // H_kv

        key_states = key_states.view(B, S, H_kv, head_dim).transpose(1, 2)   # [B, H, S, d]
        value_states = value_states.view(B, S, H_kv, head_dim).transpose(1, 2)

        key_pruned, value_pruned, _, _ = prune_kv_cache(
            key_states, value_states, keep_mask_flat, attention_mask=None
        )

        if past is not None:
            try:
                layer_idx = getattr(self.base_attn, "CoSRe_layer_idx", None)

                if layer_idx is not None:
                    # Case 1: transformers.Cache style
                    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
                        past.key_cache[layer_idx] = key_pruned
                        past.value_cache[layer_idx] = value_pruned
                    # Case 2: custom structure with _cache dict
                    elif hasattr(past, "_cache"):
                        if layer_idx in past._cache:
                            past._cache[layer_idx]["k"] = key_pruned
                            past._cache[layer_idx]["v"] = value_pruned
                    else:
                        # Unknown type; do not crash
                        logger.warning(
                            "[COSRE] EfficientAttention: unknown cache type %s; "
                            "KV not overwritten.",
                            type(past),
                        )
                else:
                    logger.warning(
                        "[COSRE] EfficientAttention: base_attn has no CoSRe_layer_idx; "
                        "cannot index into cache."
                    )
            except Exception as e:
                logger.warning(
                    "[COSRE] EfficientAttention: failed to overwrite KV cache: %s", str(e)
                )

        # return same structure
        return (attn_output, attn_weights, past)
