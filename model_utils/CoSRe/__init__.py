# model_utils/CoSRe/__init__.py

import logging
from typing import List
import torch.nn as nn

from .config import DAREConfig
from .controller import DAREController

logger = logging.getLogger(__name__)
from .config import DAREConfig
from .controller import (
    DAREController,
    ModalityRouter,
    gather_prune_kv,
    attach_CoSRe_to_anole,
)
from .attention import EfficientAttention
from .wrapped_block import DAREWrappedBlock

__all__ = [
    "DAREConfig",
    "DAREController",
    "ModalityRouter",
    "gather_prune_kv",
    "attach_CoSRe_to_anole",
    "EfficientAttention",
    "DAREWrappedBlock",
]


def _find_transformer_blocks(model: nn.Module) -> List[nn.Module]:
    cands = []

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


def attach_CoSRe_to_anole(model: nn.Module, controller: DAREController):
    """
    Wrap every Anole/Chameleon attention block in EfficientAttention and
    attach a shared DAREController.
    """
    blocks = _find_transformer_blocks(model)

    if not blocks:
        logger.warning("[DARE] Could not find Transformer blocks. DARE disabled.")
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
            attn_attr_name = None

        if isinstance(attn, nn.Module) and attn_attr_name is not None:
            # Wrap in EfficientAttention only once
            if not isinstance(attn, EfficientAttention):
                wrapped = EfficientAttention(attn)
            else:
                wrapped = attn  # already wrapped

            # Attach controller + layer index to the wrapper
            setattr(wrapped, "CoSRe_controller", controller)
            setattr(wrapped, "CoSRe_layer_idx", idx)

            # Replace original attention with the wrapper
            setattr(block, attn_attr_name, wrapped)

            num_attn += 1

    if num_attn == 0:
        logger.warning("[DARE] No attention modules found inside blocks.")
    else:
        logger.info(f"[DARE] Attached controller to {num_attn} attention layers.")

    setattr(model, "CoSRe_controller", controller)
    setattr(model, "CoSRe_enabled", True)

    return model
