"""
Linear regression utilities.

Replaces MATLAB's regress() function with numpy equivalent.

MATLAB regress():
    b = regress(y, X)
    Returns coefficients for y = X * b (ordinary least squares)

Python equivalent:
    b, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
"""

import numpy as np
from typing import Tuple, Optional


def regress(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Perform linear regression (MATLAB regress() equivalent).

    Solves the linear least squares problem: y = X * b

    Args:
        y: Response variable (N,) or (N, 1)
        X: Design matrix (N, p) where p is number of predictors.
           Typically includes a column of ones for the intercept.

    Returns:
        Coefficient vector (p,) or (p, 1) matching input shape

    Example:
        # MATLAB: b = regress(y, [ones(size(x)) x])
        # Python: b = regress(y, np.column_stack([np.ones_like(x), x]))

        # The returned b[0] is the intercept (offset)
        # The returned b[1:] are the slopes
    """
    y = np.asarray(y).flatten()
    X = np.asarray(X)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Solve least squares
    b, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    return b


def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    return_stats: bool = False
) -> Tuple[float, float] | Tuple[float, float, dict]:
    """
    Simple linear regression: y = intercept + slope * x

    Args:
        x: Independent variable (N,)
        y: Dependent variable (N,)
        return_stats: If True, also return R-squared and residuals

    Returns:
        intercept: y-intercept (offset)
        slope: slope coefficient

        If return_stats=True:
            Also returns dict with 'r_squared', 'residuals', 'y_pred'

    Example:
        # Find offset between pos*2 and pos40x coordinates
        intercept, slope = linear_regression(pos40x, pos * 2)
        # intercept is the offset we need
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} != {len(y)}")

    if len(x) < 2:
        raise ValueError(f"Need at least 2 points for regression, got {len(x)}")

    # Design matrix with intercept column
    X = np.column_stack([np.ones_like(x), x])

    # Solve
    b = regress(y, X)
    intercept = b[0]
    slope = b[1]

    if not return_stats:
        return intercept, slope

    # Calculate statistics
    y_pred = intercept + slope * x
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    stats = {
        'r_squared': r_squared,
        'residuals': residuals,
        'y_pred': y_pred,
        'ss_res': ss_res,
        'ss_tot': ss_tot,
    }

    return intercept, slope, stats


def calculate_fov_offset(
    pos: np.ndarray,
    pos40x: np.ndarray,
    scale_factor: float = 2.0
) -> Tuple[int, int]:
    """
    Calculate FOV offset using regression on cell positions.

    This is the core calculation from stitch_subslices.m:
        pos * scale_factor = offset + pos40x
        offset = intercept from regression

    Args:
        pos: Cell positions in full-resolution space (N, 2)
        pos40x: Cell positions in 40x space (N, 2)
        scale_factor: Scale factor (default 2.0)

    Returns:
        Tuple of (x_offset, y_offset) as integers

    Raises:
        ValueError: If fewer than 3 cells provided
    """
    pos = np.asarray(pos)
    pos40x = np.asarray(pos40x)

    if pos.shape[0] < 3:
        raise ValueError(f"Need at least 3 cells for regression, got {pos.shape[0]}")

    # X offset: pos(:,1)*2 = offset_x + pos40x(:,1)
    # IMPORTANT: MATLAB uses 1-indexed columns, Python uses 0-indexed
    # pos(:,1) in MATLAB = pos[:,0] in Python (x coordinate)
    # pos(:,2) in MATLAB = pos[:,1] in Python (y coordinate)
    x_offset, _ = linear_regression(pos40x[:, 0], pos[:, 0] * scale_factor)

    # Y offset
    y_offset, _ = linear_regression(pos40x[:, 1], pos[:, 1] * scale_factor)

    return round(x_offset), round(y_offset)
