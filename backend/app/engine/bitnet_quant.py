"""BitNet b1.58 ternary quantization and dynamic activation quantization engine."""
from __future__ import annotations

import math
from typing import Dict, Tuple
import numpy as np


def quantize_weights_absmean(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize floating point weights into ternary values {-1, 0, +1} using absmean scaling.

    Formula from BitNet b1.58:
        gamma = (1 / nm) * sum(|W_ij|)
        W_tilde = clip(round(W / (gamma + eps)), -1, 1)

    Returns:
        ternary_weights: ndarray of shape matching weights with values in {-1, 0, +1}
        gamma: float scaling factor
    """
    if weights.size == 0:
        return np.array([], dtype=np.int8), 1.0

    gamma = float(np.mean(np.abs(weights)))
    scale = max(gamma, 1e-7)

    scaled = weights / scale
    rounded = np.round(scaled)
    ternary = np.clip(rounded, -1, 1).astype(np.int8)

    # If all weights quantized to zero, gamma should be 1.0 for identity
    if np.all(ternary == 0):
        gamma = 1.0

    return ternary, gamma


def quantize_activations_absmax(
    activations: np.ndarray, num_bits: int = 8
) -> Tuple[np.ndarray, float]:
    """Quantize floating point activations into signed integers using dynamic absmax scaling.

    Formula:
        alpha = max(|X|)
        X_tilde = clip(round(X * (Q_b / (alpha + eps))), -Q_b, Q_b - 1)
        where Q_b = 2^(num_bits - 1) - 1 (127 for 8-bit)

    Returns:
        quantized_activations: ndarray of signed ints
        alpha: float scaling factor
    """
    if activations.size == 0:
        return np.array([], dtype=np.int8), 1.0

    q_max = (1 << (num_bits - 1)) - 1  # 127 for 8-bit
    alpha = float(np.max(np.abs(activations)))
    scale = max(alpha, 1e-7)

    scaled = activations * (q_max / scale)
    rounded = np.round(scaled)
    quantized = np.clip(rounded, -q_max - 1, q_max).astype(np.int8)

    # If all activations quantized to zero, alpha should be 1.0
    if np.all(quantized == 0):
        alpha = 1.0

    return quantized, alpha


def ternary_linear_forward(
    x: np.ndarray,
    ternary_weights: np.ndarray,
    gamma: float,
    quantize_activations: bool = True,
) -> np.ndarray:
    """Execute forward linear layer with ternary weights and optional 8-bit activations.

    y = x @ W^T
    With ternary weights, inner products become pure accumulations (add/subtract).
    """
    if quantize_activations:
        x_q, alpha = quantize_activations_absmax(x, num_bits=8)
        # Integer accumulation: int8 x int8 -> int32
        # x_q shape: [..., in_features], ternary_weights shape: [out_features, in_features]
        accum = np.matmul(x_q.astype(np.int32), ternary_weights.T.astype(np.int32))
        # Dequantize back to float
        output = accum.astype(np.float32) * (alpha * gamma / 127.0)
    else:
        # Floating point activations with ternary weights
        accum = np.matmul(x.astype(np.float32), ternary_weights.T.astype(np.float32))
        output = accum * gamma

    return output


def compute_quantization_stats(
    original_weights: np.ndarray, ternary_weights: np.ndarray, gamma: float
) -> Dict[str, float]:
    """Compute statistics including sparsity, mean squared error, and weight distribution."""
    total_elements = float(ternary_weights.size)
    if total_elements == 0:
        return {
            "sparsity": 0.0,
            "mse": 0.0,
            "weight_distribution_neg": 0.0,
            "weight_distribution_zero": 0.0,
            "weight_distribution_pos": 0.0,
            "gamma": gamma,
        }

    neg_ones = float(np.sum(ternary_weights == -1)) / total_elements
    zeros = float(np.sum(ternary_weights == 0)) / total_elements
    pos_ones = float(np.sum(ternary_weights == 1)) / total_elements

    reconstructed = ternary_weights.astype(np.float32) * gamma
    mse = float(np.mean((original_weights - reconstructed) ** 2))

    return {
        "sparsity": zeros,
        "mse": mse,
        "weight_distribution_neg": neg_ones,
        "weight_distribution_zero": zeros,
        "weight_distribution_pos": pos_ones,
        "gamma": gamma,
    }