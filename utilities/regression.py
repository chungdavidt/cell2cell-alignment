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
    scale_factor: float = 2.0,
    fit_slope: bool = True,
    return_slope: bool = False,
):
    """
    Calculate FOV offset from the cell positions inside it.

    This is the core calculation from stitch_subslices.m:
        pos * scale_factor = offset + pos40x
        offset = intercept from regression

    The model fixes the slope at 1 — the two coordinate systems differ by a
    translation and the `scale_factor` already applied — but the default fit
    leaves it free and discards it, which is why it needs 3 cells to be
    over-determined and raises below that. `fit_slope=False` pins the slope at 1
    and averages `pos*scale - pos40x` instead, so a FOV with a single cell can
    still be placed. Callers that stitch every FOV of a slice need that; step 2's
    subslice stitching keeps the default so its output is unchanged.

    Args:
        pos: Cell positions in full-resolution space (N, 2)
        pos40x: Cell positions in 40x space (N, 2)
        scale_factor: Scale factor (default 2.0)
        fit_slope: Fit and discard a slope (default, needs >= 3 cells), or pin
            it at 1 and average the residual (needs >= 1)
        return_slope: Also return the fitted (x, y) slopes — the quantity the
            default path throws away, and the check on whether pinning it costs
            anything. Slopes are None when fit_slope=False.

    Returns:
        (x_offset, y_offset) as integers, or
        (x_offset, y_offset, (x_slope, y_slope)) when return_slope is True

    Raises:
        ValueError: If too few cells for the chosen estimator
    """
    pos = np.asarray(pos)
    pos40x = np.asarray(pos40x)

    if fit_slope:
        if pos.shape[0] < 3:
            raise ValueError(f"Need at least 3 cells for regression, got {pos.shape[0]}")

        # X offset: pos(:,1)*2 = offset_x + pos40x(:,1)
        # IMPORTANT: MATLAB uses 1-indexed columns, Python uses 0-indexed
        # pos(:,1) in MATLAB = pos[:,0] in Python (x coordinate)
        # pos(:,2) in MATLAB = pos[:,1] in Python (y coordinate)
        x_offset, x_slope = linear_regression(pos40x[:, 0], pos[:, 0] * scale_factor)

        # Y offset
        y_offset, y_slope = linear_regression(pos40x[:, 1], pos[:, 1] * scale_factor)
    else:
        if pos.shape[0] < 1:
            raise ValueError("Need at least 1 cell to place a FOV, got 0")

        x_offset = float(np.mean(pos[:, 0] * scale_factor - pos40x[:, 0]))
        y_offset = float(np.mean(pos[:, 1] * scale_factor - pos40x[:, 1]))
        x_slope = y_slope = None

    if return_slope:
        return round(x_offset), round(y_offset), (x_slope, y_slope)
    return round(x_offset), round(y_offset)
