"""Centered Kernel Alignment (CKA) for metric bundle Layer 2.

Linear CKA per Kornblith et al. (2019) ICML.
Measures similarity between two representation matrices.
"""

import hashlib

import numpy as np

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between two representation matrices.

    X, Y: (n_samples, n_features) or (n_features, n_features).
    If square matrices (weight matrices), rows are treated as samples.
    Returns scalar in [0, 1]. 1.0 = identical representations.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)

def snapshot_hash(matrix: np.ndarray) -> str:
    """SHA256 hash of a numpy array for reproducibility verification."""
    return hashlib.sha256(matrix.tobytes()).hexdigest()[:16]
