"""Controller network u_theta used by DBPS and the learned baselines.

DBPS (cond_on_y_star=False): u_theta(x_t, t [, age])
  Input:  [x_t || time feature || age feature?]
  Output: u of shape [batch, d]
  At sampling time the DBPS controller is built externally as
    (1 - rho) * compute_u_anchor(x_t, t, y*) + rho * network(x_t, t [, age]).
cond_on_y_star=True additionally concatenates the anchor y* to the input.

Options (set from the config):
  activation: "silu" | "relu"
  time_embedding_type: "sinusoidal" | "raw" (scalar t concatenated)
  block_type: "residual" | "plain" (Linear + activation, no skip connections)
  cond_on_age: when True the network also consumes the source subject's
    age, normalised as (age - 70) / 10 and optionally embedded.

Network code uses torch.float32 by default (training-friendly); inputs
arriving in float64 are cast at the boundary inside forward().
"""
from typing import Optional

import torch
import torch.nn as nn

from models.mlp_blocks import ResidualMLPBlock, SinusoidalTimeEmbedding


def _activation_fn(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name!r}")


class _PlainBlock(nn.Module):
    """Linear + activation, no skip / norm."""

    def __init__(self, hidden_dim: int, activation: str, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = _activation_fn(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.fc(x)))


class ControllerNet(nn.Module):
    """MLP body conditioned on x_t, time, optionally y*."""

    def __init__(
        self,
        d: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        cond_on_y_star: bool = False,
        cond_on_age: bool = False,
        age_emb_dim: int = 1,
        time_emb_dim: int = 32,
        dropout: float = 0.0,
        activation: str = "silu",
        time_embedding_type: str = "sinusoidal",
        block_type: str = "residual",
    ):
        super().__init__()

        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cond_on_y_star = cond_on_y_star
        self.cond_on_age = cond_on_age
        self.age_emb_dim = int(age_emb_dim)
        self.activation = activation
        self.time_embedding_type = time_embedding_type
        self.block_type = block_type
        # Age head: scalar (age-70)/10 when age_emb_dim == 1, else a
        # learnable Linear(1, age_emb_dim) -> ReLU embedding.
        if cond_on_age and self.age_emb_dim > 1:
            self.age_emb = nn.Sequential(
                nn.Linear(1, self.age_emb_dim),
                _activation_fn(activation),
            )
        else:
            self.age_emb = None

        if time_embedding_type == "sinusoidal":
            if time_emb_dim % 2 != 0:
                raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}")
            self.time_emb_dim = time_emb_dim
            self.time_emb: Optional[nn.Module] = SinusoidalTimeEmbedding(time_emb_dim)
            t_feat_dim = time_emb_dim
        elif time_embedding_type == "raw":
            # Concatenate scalar t, no embedding.
            self.time_emb_dim = 1
            self.time_emb = None
            t_feat_dim = 1
        else:
            raise ValueError(
                f"time_embedding_type must be 'sinusoidal' or 'raw', "
                f"got {time_embedding_type!r}"
            )

        input_dim = (
            d                                    # x_t
            + t_feat_dim                         # t (raw or sinusoidal)
            + (d if cond_on_y_star else 0)       # y*
            + (self.age_emb_dim if cond_on_age else 0)   # age (scalar or embedded)
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = _activation_fn(activation)
        if block_type == "residual":
            # ResidualMLPBlock's activation is fixed to SiLU; for other
            # activations we fall back to plain blocks.
            if activation.lower() == "silu":
                self.body = nn.Sequential(
                    *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)],
                )
            else:
                self.body = nn.Sequential(
                    *[_PlainBlock(hidden_dim, activation, dropout) for _ in range(num_layers)],
                )
        elif block_type == "plain":
            self.body = nn.Sequential(
                *[_PlainBlock(hidden_dim, activation, dropout) for _ in range(num_layers)],
            )
        else:
            raise ValueError(
                f"block_type must be 'residual' or 'plain', got {block_type!r}"
            )
        self.output_proj = nn.Linear(hidden_dim, d)

    def _t_feature(self, t_f: torch.Tensor) -> torch.Tensor:
        """Convert t to feature vector based on time_embedding_type."""
        if self.time_embedding_type == "sinusoidal":
            return self.time_emb(t_f)
        # raw: ensure shape [B, 1]
        if t_f.ndim == 1:
            return t_f.unsqueeze(-1)
        if t_f.ndim == 2 and t_f.shape[-1] == 1:
            return t_f
        raise ValueError(f"raw t expects shape [B] or [B, 1]; got {t_f.shape}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y_star: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_t_f = x_t.to(torch.float32)
        t_f = t.to(torch.float32)
        time_feat = self._t_feature(t_f)

        parts = [x_t_f, time_feat]
        if self.cond_on_y_star:
            assert y_star is not None, (
                "ControllerNet was constructed with cond_on_y_star=True; "
                "y_star must be provided."
            )
            parts.append(y_star.to(torch.float32))
        if self.cond_on_age:
            assert age is not None, (
                "ControllerNet was constructed with cond_on_age=True; "
                "age must be provided."
            )
            age_f = age.to(torch.float32)
            if age_f.ndim == 1:
                age_f = age_f.unsqueeze(-1)            # [B] -> [B, 1]
            # Center+scale: age values ~60-95 dominate input proj if raw.
            age_norm = (age_f - 70.0) / 10.0
            if self.age_emb is not None:               # learned embedding
                age_feat = self.age_emb(age_norm)      # [B, age_emb_dim]
            else:                                       # scalar
                age_feat = age_norm                     # [B, 1]
            parts.append(age_feat)
        # When a flag is False, the corresponding kwarg is silently ignored
        # so a single calling code path can pass y_star / age regardless of
        # the network type.
        inp = torch.cat(parts, dim=-1)

        h = self.input_act(self.input_proj(inp))
        h = self.body(h)
        return self.output_proj(h)
