"""Building blocks for ControllerNet.

NOTE on dtype: networks use torch.float32 by default (standard for
training). All math/control-law code in controls/ and dynamics/ uses
torch.float64. When the network output is consumed by the rollout
it is cast to float64 at the boundary. This is
documented at the call site in sampling/ and training/.
"""
import math
from typing import Optional, Type

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding: t -> [sin, cos] over a range
    of frequencies. Output dim must be even.

    Args:
      dim: output embedding dimension. Must be even.

    Forward:
      t: [batch] in [0, 1] (any float dtype, internally cast to float32).
      Returns [batch, dim] in float32.

    For j in 0..dim/2-1, with freq_j = 1 / 10000**(2*j / dim):
      emb[b, 2*j]   = sin(t[b] * freq_j)
      emb[b, 2*j+1] = cos(t[b] * freq_j)

    The frequencies are computed via torch.exp(linspace) for numerical
    stability and registered as a non-persistent buffer.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding dim must be even, got {dim}")
        self.dim = dim

        half = dim // 2
        # log(10000) decay across the half-dim positions; geometric spacing.
        freqs = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), half, dtype=torch.float32)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_f = t.to(torch.float32)
        if t_f.ndim != 1:
            raise ValueError(
                f"SinusoidalTimeEmbedding expects t shape [batch], got {tuple(t_f.shape)}"
            )
        scaled = t_f.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [batch, half]
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)
        # Interleave: emb[b, 2j] = sin, emb[b, 2j+1] = cos.
        emb = torch.stack([sin_part, cos_part], dim=-1)       # [batch, half, 2]
        return emb.reshape(t_f.shape[0], self.dim)            # [batch, dim]


class ResidualMLPBlock(nn.Module):
    """Linear -> activation -> Dropout -> Linear -> Residual.

    Args:
      hidden_dim: input/output dim (matched for the residual sum).
      dropout: dropout probability (0.0 disables).
      activation: nn.Module class used as activation factory; defaults to nn.SiLU.

    Forward:
      x: [batch, hidden_dim] -> [batch, hidden_dim].
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.SiLU
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(x))
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h
