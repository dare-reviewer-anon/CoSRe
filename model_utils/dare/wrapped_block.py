from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from .controller import DAREController


class DAREWrappedBlock(nn.Module):
    """
    Wraps a decoder block with DARE routing.

    Assumes `base_block` has:
      - self_attn
      - mlp
      - input/output layer norms ln1, ln2
    """

    def __init__(self, base_block: nn.Module, dare_controller: DAREController, layer_idx: int):
        super().__init__()
        self.block = base_block
        self.dare = dare_controller
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, T, D]
        modality_mask: torch.Tensor,          # [B, T]
        hop_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        training: bool = True,
    ):
        attn_out, new_past_kv = self.block.self_attn(
            self.block.ln1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out

        mlp_out = self.block.mlp(self.block.ln2(hidden_states))
        hidden_states = hidden_states + mlp_out

        dare_out = self.router(
            hidden_states,
            modality_mask=modality_mask,
            hop_idx=hop_idx,
            layer_idx=self.layer_idx,
            training=training,
        )
        keep_mask = dare_out["keep_mask"]
        exec_mask = dare_out["exec_mask"]

        # pass keep_mask into attention
        attn_out, new_past_kv = self.block.self_attn(
            self.block.ln1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            keep_mask=keep_mask,
        )

        hidden_states = hidden_states * exec_mask.unsqueeze(-1)

        aux_losses: Dict[str, torch.Tensor] = {
            "L_text_ratio": dare_out["L_text_ratio"],
            "L_vis_soft": dare_out["L_vis_soft"],
            "L_vis_hard": dare_out["L_vis_hard"],
        }

        return hidden_states, new_past_kv, exec_mask, aux_losses
