from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .controller import CoSReController, CoSReControllerConfig



@dataclass
class CoSReHookConfig:
    enable: bool = False
    image_hw: Tuple[int, int] = (32, 32)

    strict_image_mask: bool = True

    force_use_cache: bool = True


ImageMaskFn = Callable[..., torch.Tensor]


def _default_image_mask_fn(*args, **kwargs) -> torch.Tensor:
    raise RuntimeError(
        "CoSRe image_mask is required but no image_mask_fn was provided. "
        "Pass image_mask=... to generate(), or provide image_mask_fn."
    )


def _make_attention_mask_like(prefix_embeds: torch.Tensor) -> torch.Tensor:
    # prefix_embeds: (B, Lp, d)
    B, Lp, _ = prefix_embeds.shape
    return torch.ones((B, Lp), device=prefix_embeds.device, dtype=torch.long)


def build_cosre_prefix_embeds(
    model: nn.Module,
    controller: CoSReController,
    input_ids: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_hw: Optional[Tuple[int, int]] = None,
    return_aux: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:

    if not hasattr(model, "get_input_embeddings"):
        raise ValueError("Model must implement get_input_embeddings() to build inputs_embeds.")

    emb_layer = model.get_input_embeddings()
    embeds = emb_layer(input_ids)  # (B, L, d)

    prefix = controller.build_memory_prefix_from_inputs(
        model=model,
        input_ids=input_ids,
        image_mask=image_mask,
        attention_mask=attention_mask,
        grid_hw=image_hw if image_hw is not None else controller.cfg.image_hw,
        return_aux=return_aux,
    )
    return prefix




class CoSReGenerationHook:


    def __init__(
        self,
        controller: CoSReController,
        cfg: CoSReHookConfig,
        image_mask_fn: Optional[ImageMaskFn] = None,
    ):
        super().__init__()
        self.controller = controller
        self.cfg = cfg
        self.image_mask_fn = image_mask_fn or _default_image_mask_fn

    @classmethod
    def attach(
        cls,
        model: nn.Module,
        enable: bool,
        image_hw: Tuple[int, int],
        image_mask_fn: Optional[ImageMaskFn] = None,
        controller_cfg: Optional[CoSReControllerConfig] = None,
        strict_image_mask: bool = True,
        force_use_cache: bool = True,
    ) -> "CoSReGenerationHook":
        """
        Build a hook + controller from model.
        """
        if controller_cfg is None:
            controller_cfg = CoSReControllerConfig(image_hw=image_hw)
        controller = CoSReController.from_model(model, cfg=controller_cfg)
        hook_cfg = CoSReHookConfig(
            enable=enable,
            image_hw=image_hw,
            strict_image_mask=strict_image_mask,
            force_use_cache=force_use_cache,
        )
        return cls(controller=controller, cfg=hook_cfg, image_mask_fn=image_mask_fn)

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        
        generation_kwargs = generation_kwargs or {}

        if not self.cfg.enable:
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
            return (out, {}) if return_aux else out

        # Ensure we have image_mask
        if image_mask is None:
            try:
                image_mask = self.image_mask_fn(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
            except Exception as e:
                if self.cfg.strict_image_mask:
                    raise
                # fallback to normal generate
                out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
                return (out, {"cosre_disabled_reason": f"image_mask_missing: {e}"}) if return_aux else out

        if image_mask.dtype != torch.bool:
            image_mask = image_mask.to(dtype=torch.bool)

        # CoSRe is a KV-cache method: for generation we want cache ON.
        use_cache_prev = None
        if self.cfg.force_use_cache and hasattr(model, "config") and hasattr(model.config, "use_cache"):
            use_cache_prev = model.config.use_cache
            model.config.use_cache = True

        # Build CoSRe prefix embeddings
        prefix, aux = self.controller.build_memory_prefix_from_inputs(
            model=model,
            input_ids=input_ids,
            image_mask=image_mask,
            attention_mask=attention_mask,
            grid_hw=self.cfg.image_hw,
            return_aux=True,
        )

        # Build attention mask for prefix
        prefix_attn = _make_attention_mask_like(prefix)

        out = model.generate(inputs_embeds=prefix, attention_mask=prefix_attn, **generation_kwargs)

        # Restore use_cache if we changed it
        if use_cache_prev is not None:
            model.config.use_cache = use_cache_prev

        if return_aux:
            aux_out: Dict[str, Any] = {"cosre": aux, "prefix_len": prefix.shape[1]}
            return out, aux_out
        return out




def attach_cosre_generate_to_model(
    model: nn.Module,
    hook: CoSReGenerationHook,
    method_name: str = "cosre_generate",
) -> nn.Module:

    if hasattr(model, method_name):
        # Don't silently overwrite; consistent & safe.
        raise AttributeError(f"Model already has attribute '{method_name}'.")

    def _cosre_generate(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        return_aux: bool = False,
    ):
        return hook.generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_mask=image_mask,
            generation_kwargs=generation_kwargs,
            return_aux=return_aux,
        )

    setattr(model, method_name, _cosre_generate)
    return model


