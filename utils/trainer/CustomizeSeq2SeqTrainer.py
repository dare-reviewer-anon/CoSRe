# utils/trainer/customize_trainer.py

from __future__ import annotations

import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput


@torch.no_grad()
def greedy_decode_with_prefix(
    model,
    prefix_embeds: torch.Tensor,
    max_new_tokens: int,
    stopping_criteria=None,
) -> torch.Tensor:
    
    # 1) Build KV cache from prefix embeddings
    out = model(inputs_embeds=prefix_embeds, use_cache=True)
    past = out.past_key_values

    # 2) Start token (decoder_start_token_id preferred, else BOS)
    if getattr(model.config, "decoder_start_token_id", None) is not None:
        cur = torch.tensor([[model.config.decoder_start_token_id]], device=prefix_embeds.device)
    else:
        bos = getattr(model.config, "bos_token_id", None)
        if bos is None:
            raise ValueError("Model has neither decoder_start_token_id nor bos_token_id.")
        cur = torch.tensor([[bos]], device=prefix_embeds.device)

    generated: List[torch.Tensor] = []

    for _ in range(int(max_new_tokens)):
        o = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]  # (1, vocab)
        nxt = torch.argmax(logits, dim=-1, keepdim=True)  # (1,1)
        generated.append(nxt)

        # HF stopping criteria expects "input_ids" (the full generated sequence)
        if stopping_criteria is not None and len(generated) > 0:
            sofar = torch.cat(generated, dim=1)  # (1,t)
            # stopping_criteria returns True when we should stop
            if stopping_criteria(sofar, None):
                break

        # stop on EOS
        eos = getattr(model.config, "eos_token_id", None)
        if eos is not None and int(nxt.item()) == int(eos):
            break

        cur = nxt

    if len(generated) == 0:
        return torch.empty((1, 0), dtype=torch.long, device=prefix_embeds.device)
    return torch.cat(generated, dim=1)


class CustomizeSeq2SeqTrainer(Seq2SeqTrainer):
    
    def image_token_mask_fn(self, input_ids_1d: torch.Tensor, **kwargs) -> torch.Tensor:
        
        L = input_ids_1d.numel()
        n_img = getattr(self.args, "image_seq_length", None)
        if n_img is None:
            raise ValueError("args.image_seq_length is required for default image_token_mask_fn.")
        if n_img > L:
            raise ValueError(f"image_seq_length={n_img} > seq_len={L}.")
        mask = torch.zeros((L,), dtype=torch.bool, device=input_ids_1d.device)
        mask[:n_img] = True
        return mask

    
    def _get_cosre_controller(self):
        """
        Lazily create a CoSRe controller. Assumes you placed your modules at:
          model_utils/cosre/controller.py
        """
        if hasattr(self, "_cosre_controller") and self._cosre_controller is not None:
            return self._cosre_controller

        from model_utils.cosre.controller import CoSReController, CoSReControllerConfig

        # Infer visual grid (H,W) from image_seq_length
        n_img = int(getattr(self.args, "image_seq_length", 1024))
        H = int(round(n_img ** 0.5))
        W = H
        if H * W != n_img:
            # If not a perfect square, you must specify image_hw explicitly somewhere
            raise ValueError(
                f"image_seq_length={n_img} is not a square. "
                "Please set controller cfg.image_hw explicitly."
            )

        cfg = CoSReControllerConfig(
            image_hw=(H, W),
            codec_block_size=int(getattr(self.args, "cosre_block", 8)),
            codec_keep_hw=(int(getattr(self.args, "cosre_keep_h", 4)), int(getattr(self.args, "cosre_keep_w", 4))),
            codec_base_delta=float(getattr(self.args, "cosre_base_delta", 0.5)),
            codec_delta_mode=str(getattr(self.args, "cosre_delta_mode", "linear")),
            codec_hard_round=bool(getattr(self.args, "cosre_hard_round", True)),
            num_slots=int(getattr(self.args, "cosre_slots", 32)),
            num_heads=int(getattr(self.args, "cosre_heads", 8)),
            fusion_mode=str(getattr(self.args, "cosre_fusion_mode", "scalar")),
        )
        controller = CoSReController.from_model(self.model, cfg=cfg).to(self.model.device)
        self._cosre_controller = controller
        return controller

    @torch.no_grad()
    def _cosre_generate_batch(
        self,
        inputs: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        stopping_criteria=None,
    ) -> torch.Tensor:
        
        model = self.model
        device = next(model.parameters()).device

        input_ids = inputs["input_ids"].to(device)
        B, L = input_ids.shape

        emb_layer = model.get_input_embeddings()
        controller = self._get_cosre_controller()

        # max_new_tokens resolution:
        max_new = generation_kwargs.get("max_new_tokens", None)
        if max_new is None:
            max_new = getattr(self.args, "generation_max_new_tokens", 256)
        max_new = int(max_new)

        outputs: List[torch.Tensor] = []
        for b in range(B):
            ids = input_ids[b]  # (L,)
            mask_img = self.image_token_mask_fn(ids, inputs=inputs, index=b)  # (L,) bool

            embs = emb_layer(ids)  # (L,d)
            vis_embs = embs[mask_img]      # (Nv,d)
            text_embs = embs[~mask_img]    # (Nt,d)

            # controller expects batch dims (B=1)
            text_embs_b = text_embs.unsqueeze(0)  # (1,Nt,d)
            vis_embs_b = vis_embs.unsqueeze(0)    # (1,Nv,d)

            prefix = controller.build_memory_prefix(
                text_tokens=text_embs_b,
                vis_tokens=vis_embs_b,
                grid_hw=controller.cfg.image_hw,
                return_aux=False,
            )  # (1,Lm,d)

            gen_ids = greedy_decode_with_prefix(
                model=model,
                prefix_embeds=prefix,
                max_new_tokens=max_new,
                stopping_criteria=stopping_criteria,
            )  # (1,T)
            outputs.append(gen_ids.squeeze(0))

        # Pad to (B, T_max)
        if len(outputs) == 0:
            return torch.empty((0, 0), dtype=torch.long, device=device)

        pad_id = getattr(model.config, "pad_token_id", None)
        if pad_id is None:
            # fallback: eos or 0
            pad_id = getattr(model.config, "eos_token_id", 0)

        T_max = max(int(x.numel()) for x in outputs)
        out = torch.full((B, T_max), fill_value=int(pad_id), dtype=torch.long, device=device)
        for i, seq in enumerate(outputs):
            if seq.numel() > 0:
                out[i, : seq.numel()] = seq
        return out


    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys
        )

        if prediction_loss_only:
            return (loss, None, None)

        enable_cosre = bool(getattr(self.args, "enable_CoSRe", False))
        if not enable_cosre:
            # fall back to normal HF Seq2SeqTrainer generation path
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        # Collect generation kwargs from self._gen_kwargs if available, else config defaults.
        gen_kwargs = getattr(self, "_gen_kwargs", {}) or {}
        if "max_new_tokens" not in gen_kwargs and hasattr(self.args, "generation_max_new_tokens"):
            gen_kwargs["max_new_tokens"] = int(self.args.generation_max_new_tokens)

        stopping_criteria = getattr(self.args, "customize_gen_stopping_criteria", None)

        generated_tokens = self._cosre_generate_batch(
            inputs=inputs,
            generation_kwargs=gen_kwargs,
            stopping_criteria=stopping_criteria,
        )

        if "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None

        # Return (loss, generated_tokens, labels) in the exact format Trainer expects
        return (loss, generated_tokens, labels)
