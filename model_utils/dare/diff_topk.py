import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableTopK(nn.Module):
    """
    Differentiable Top-K using Gumbel-softmax relaxation with straight-through.
    Training: soft mask, gradient flows.
    Inference: hard mask.
    """
    def __init__(self, temperature: float = 0.5, straight_through: bool = True):
        super().__init__()
        self.temperature = temperature
        self.straight_through = straight_through

    def forward(self, scores: torch.Tensor, k: int, training: bool = True):
        B, T = scores.shape

        # K >= T -> keep everything
        if k >= T:
            return scores.new_ones(B, T)

        # Inference -> pure hard top-k
        if not training:
            thresh = torch.topk(scores, k, dim=1).values[:, -1].unsqueeze(1)
            hard = (scores >= thresh).float()
            return hard

        # ---- Gumbel sampling ----
        g = -torch.empty_like(scores).exponential_().log()
        logits = (scores + g) / max(self.temperature, 1e-5)
        probs = F.softmax(logits, dim=1)   # (B, T)

        # Scale sum to ~ k
        scale = k / (probs.sum(dim=1, keepdim=True) + 1e-9)
        soft = (probs * scale).clamp(max=1.0)

        # Straight-through
        if self.straight_through:
            with torch.no_grad():
                thresh = torch.topk(scores, k, dim=1).values[:, -1].unsqueeze(1)
                hard = (scores >= thresh).float()
            return soft + (hard - soft).detach()
        else:
            return soft
