"""Training loss for the DBPS controller."""
import torch


def mse_controller_loss(
    u_pred: torch.Tensor, u_label: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted controller and target label.

    Args:
      u_pred:  [B, d] from network forward (typically float32).
      u_label: [B, d] from compute_u_pop_label
               (typically float64).

    Both tensors are cast to float64 internally; returns a scalar
    float64 loss. Gradient flows through the cast back to the
    network's float32 parameters.
    """
    p = u_pred.to(torch.float64)
    l = u_label.to(torch.float64)
    if p.shape != l.shape:
        raise ValueError(
            f"mse_controller_loss: shape mismatch u_pred {tuple(p.shape)} "
            f"vs u_label {tuple(l.shape)}"
        )
    return torch.mean((p - l) ** 2)
