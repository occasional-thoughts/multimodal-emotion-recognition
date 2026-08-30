"""System B — acoustic encoder.
PyTorch port of TIM-Net (Ye et al., ICASSP 2023) so it can train end-to-end
inside the PyTorch fusion model (Rajan et al.). Original is TF/Keras.

Input :  (batch, time, feat_dim)   feat_dim = 39 (MFCC) or 52 (MFCC+GFCC)
Output:  seq    (batch, n_levels, nb_filters)  <- sequence for cross-attention
         pooled (batch, nb_filters)            <- fixed vector
"""
import torch, torch.nn as nn


class CausalConv1d(nn.Module):
    """Keras padding='causal': pad only on the left so no future leakage."""
    def __init__(self, cin, cout, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel_size, dilation=dilation)

    def forward(self, x):                      # x: (B, C, T)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TemporalAwareBlock(nn.Module):
    """Two dilated causal convs, then a SIGMOID GATE multiplied onto the
    residual path (note: multiplicative gate, not an additive residual)."""
    def __init__(self, cin, nb_filters, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(cin, nb_filters, kernel_size, dilation)
        self.b1 = nn.BatchNorm1d(nb_filters)
        self.c2 = CausalConv1d(nb_filters, nb_filters, kernel_size, dilation)
        self.b2 = nn.BatchNorm1d(nb_filters)
        self.act = nn.ReLU()
        self.drop = nn.Dropout1d(dropout)      # channel-wise = Keras SpatialDropout1D
        self.match = nn.Conv1d(cin, nb_filters, 1) if cin != nb_filters else None

    def forward(self, x):
        res = x
        h = self.drop(self.act(self.b1(self.c1(x))))
        h = self.drop(self.act(self.b2(self.c2(h))))
        if self.match is not None:
            res = self.match(res)
        return res * torch.sigmoid(h)          # gate


class TIMNetEncoder(nn.Module):
    def __init__(self, feat_dim=39, nb_filters=39, kernel_size=2,
                 dilations=8, dropout=0.1, n_classes=None):
        super().__init__()
        self.dilations = dilations
        self.fwd_in = CausalConv1d(feat_dim, nb_filters, 1, 1)
        self.bwd_in = CausalConv1d(feat_dim, nb_filters, 1, 1)
        rates = [2 ** i for i in range(dilations)]
        self.fwd = nn.ModuleList(
            [TemporalAwareBlock(nb_filters, nb_filters, kernel_size, r, dropout) for r in rates])
        self.bwd = nn.ModuleList(
            [TemporalAwareBlock(nb_filters, nb_filters, kernel_size, r, dropout) for r in rates])
        # learned weighting over the n_levels scales (Keras WeightLayer)
        self.level_w = nn.Parameter(torch.rand(dilations, 1) * 0.05)
        self.head = nn.Linear(nb_filters, n_classes) if n_classes else None

    def forward(self, x):                      # x: (B, T, F)
        x = x.transpose(1, 2)                  # -> (B, F, T)
        f = self.fwd_in(x)
        b = self.bwd_in(torch.flip(x, dims=[2]))   # reversed in time
        levels = []
        for fb, bb in zip(self.fwd, self.bwd):
            f, b = fb(f), bb(b)
            levels.append((f + b).mean(dim=2))     # GlobalAveragePooling1D
        seq = torch.stack(levels, dim=1)           # (B, n_levels, nb_filters)
        pooled = (seq.transpose(1, 2) @ self.level_w).squeeze(-1)   # (B, nb_filters)
        if self.head is not None:
            return seq, pooled, self.head(pooled)
        return seq, pooled


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, fdim, dil in [("MFCC only", 39, 8), ("MFCC+GFCC", 52, 8), ("IEMOCAP cfg", 52, 10)]:
        m = TIMNetEncoder(feat_dim=fdim, dilations=dil, n_classes=7)
        x = torch.randn(4, 300, fdim)
        seq, pooled, logits = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:12} in{tuple(x.shape)} -> seq{tuple(seq.shape)} pooled{tuple(pooled.shape)} "
              f"logits{tuple(logits.shape)}  params={n:,}")
    # gradient check
    loss = logits.sum(); loss.backward()
    g = m.fwd[0].c1.conv.weight.grad
    print(f"\nbackward OK  grad_norm={g.norm():.4f}")
