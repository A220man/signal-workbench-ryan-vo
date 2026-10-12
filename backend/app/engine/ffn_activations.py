"""Activation functions for BitNet Feed-Forward Networks: Squared ReLU vs SiLU."""
from __future__ import annotations

from typing import Dict
import numpy as np


def relu_squared(x: np.ndarray) -> np.ndarray:
    """Squared ReLU activation: relu(x)^2.

    Official activation function for BitNet b1.58 architecture.
    Provides strict non-negativity with quadratic growth that preserves
    gradient scale across deep 1-bit quantized layers.
    """
    relu_val = np.maximum(0.0, x)
    return relu_val * relu_val


def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).

    Erroneously used in early llama.cpp BitNet ports (Issue #588),
    causing activation degradation into the sub-norm and perplexity regression (99.8 vs 17.1).
    """
    # Numerically stable sigmoid
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return x * sigmoid


def ffn_sublayer(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    activation: str = "relu_sqr",
) -> np.ndarray:
    """Compute the Feed-Forward Network sublayer for BitNet b1.58.

    Architecture:
        gate = act_fn(x @ W_gate^T)
        up   = x @ W_up^T
        out  = (gate * up) @ W_down^T
    """
    gate_linear = np.matmul(x, w_gate.T)
    up = np.matmul(x, w_up.T)

    if activation == "relu_sqr":
        act = relu_squared(gate_linear)
    elif activation == "silu":
        act = silu(gate_linear)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    hidden = act * up
    out = np.matmul(hidden, w_down.T)
    return out


def compare_activation_energy(x: np.ndarray) -> Dict[str, float]:
    """Measure activation norms and sparsity comparing ReLU² and SiLU.

    This directly diagnoses the calibration failure reported in upstream Issue #588:
    SILU allows negative activations and fails to scale gate outputs properly,
    diminishing activation energy over 30 layers.
    """
    act_relu_sqr = relu_squared(x)
    act_silu = silu(x)

    norm_relu_sqr = float(np.linalg.norm(act_relu_sqr))
    norm_silu = float(np.linalg.norm(act_silu))

    sparsity_relu_sqr = float(np.mean(act_relu_sqr == 0.0))
    sparsity_silu = float(np.mean(np.abs(act_silu) < 1e-6))

    mean_relu_sqr = float(np.mean(act_relu_sqr))
    mean_silu = float(np.mean(act_silu))

    return {
        "relu_sqr_l2_norm": norm_relu_sqr,
        "silu_l2_norm": norm_silu,
        "relu_sqr_sparsity": sparsity_relu_sqr,
        "silu_sparsity": sparsity_silu,
        "relu_sqr_mean": mean_relu_sqr,
        "silu_mean": mean_silu,
        "energy_ratio_silu_to_relu_sqr": (norm_silu / norm_relu_sqr) if norm_relu_sqr > 0 else 0.0,
    }
