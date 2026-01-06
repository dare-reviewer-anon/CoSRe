# utils/cosre.py
import torch
import torch.nn as nn
import math

def build_dct_matrix(N: int, device=None, dtype=None):
    n = torch.arange(N, device=device, dtype=dtype).view(1, -1)
    k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
    mat = torch.cos(math.pi * (2*n + 1) * k / (2*N))
    mat[0, :] *= math.sqrt(1.0 / N)
    mat[1:, :] *= math.sqrt(2.0 / N)
    return mat

class CoSReCompress(nn.Module):
    """
    Compress visual-token grid embeddings + build shared/residual memory tokens.
    Input:  text_embs  (Nt, d)
            vis_embs   (Nv, d) where Nv=H*W
    Output: Xs         (K + Nt + Nv_c, d)
    """
    def __init__(self, d, image_hw=(32, 32), block=8, keep_hw=(4, 4), K=32, nheads=8, base_delta=0.5):
        super().__init__()
        H, W = image_hw
        assert H % block == 0 and W % block == 0
        self.H, self.W = H, W
        self.b = block
        self.kh, self.kw = keep_hw
        self.K = K
        self.d = d

        # DCT
        D = build_dct_matrix(block)
        self.register_buffer("D", D)

        # Omega as top-left kh×kw rectangle in frequency plane
        Omega = torch.zeros(block, block)
        Omega[:self.kh, :self.kw] = 1.0
        self.register_buffer("Omega", Omega)

        # Delta table: larger steps for higher freq
        # (simple & stable default; you can tune)
        u = torch.arange(block).view(-1, 1).float()
        v = torch.arange(block).view(1, -1).float()
        Delta = base_delta * (1.0 + u + v)  # (b,b)
        self.register_buffer("Delta", Delta)

        # Projections and slot attention
        self.WT = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.slots = nn.Parameter(torch.randn(K, d) * 0.02)

        self.attn_slots_T = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.attn_slots_V = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.attn_explain_T = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.attn_explain_V = nn.MultiheadAttention(d, nheads, batch_first=True)

        self.ln_S = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _bct2d(self, X):  # (B,b,b,d)
        D = self.D
        X = X.permute(0, 3, 1, 2)           # (B,d,b,b)
        X = torch.matmul(D, X)             # (b,b)@(B,d,b,b)
        X = torch.matmul(X, D.t())
        return X.permute(0, 2, 3, 1)       # (B,b,b,d)

    def _ibct2d(self, X):  # (B,b,b,d)
        D = self.D
        X = X.permute(0, 3, 1, 2)
        X = torch.matmul(D.t(), X)
        X = torch.matmul(X, D)
        return X.permute(0, 2, 3, 1)

    @torch.no_grad()
    def compress_visual(self, vis_embs):
        """
        vis_embs: (Nv,d) with Nv=H*W
        returns:  Vs (Nv_c, d)  compressed visual tokens
        """
        H, W, d = self.H, self.W, self.d
        b = self.b
        x = vis_embs.view(H, W, d)

        # blockify: (Hb,Wb,b,b,d) -> (Bn,b,b,d)
        x = x.view(H // b, b, W // b, b, d).permute(0, 2, 1, 3, 4).contiguous()
        Bn = (H // b) * (W // b)
        blocks = x.view(Bn, b, b, d)

        C = self._bct2d(blocks)
        Q = torch.round(C / self.Delta[..., None])
        Q = Q * self.Omega[..., None]
        rec = self._ibct2d(Q)

        # keep only spatial kh×kw per block (corresponding to low-freq rectangle)
        rec = rec[:, :self.kh, :self.kw, :]  # (Bn,kh,kw,d)

        # unblock => (H',W',d) where H'=(H/b)*kh
        rec = rec.view(H // b, W // b, self.kh, self.kw, d).permute(0, 2, 1, 3, 4).contiguous()
        Hh, Wh = (H // b) * self.kh, (W // b) * self.kw
        grid = rec.view(Hh, Wh, d)
        return grid.view(-1, d)  # (Nv_c,d)

    def forward(self, text_embs, vis_embs):
        """
        text_embs: (Nt,d)
        vis_embs:  (Nv,d)
        """
        Vs = self.compress_visual(vis_embs)

        Tt = self.WT(text_embs).unsqueeze(0)  # (1,Nt,d)
        Vt = self.WV(Vs).unsqueeze(0)         # (1,Nvc,d)
        slots = self.slots.unsqueeze(0)       # (1,K,d)

        ST, _ = self.attn_slots_T(slots, Tt, Tt)
        SV, _ = self.attn_slots_V(slots, Vt, Vt)

        a = torch.sigmoid(self.alpha)
        S = self.ln_S(a * ST + (1 - a) * SV)  # (1,K,d)

        That, _ = self.attn_explain_T(Tt, S, S)
        Vhat, _ = self.attn_explain_V(Vt, S, S)

        RT = (Tt - That).squeeze(0)
        RV = (Vt - Vhat).squeeze(0)
        Ss = S.squeeze(0)

        Xs = torch.cat([Ss, RT, RV], dim=0)  # (K+Nt+Nvc, d)
        return Xs
