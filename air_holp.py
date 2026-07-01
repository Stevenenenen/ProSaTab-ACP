# air_holp.py
# -*- coding: utf-8 -*-
"""
Air-HOLP (Adaptive Iterative Ridge-HOLP) - Python implementation
Designed to closely follow the R reference code you pasted and the Air-HOLP paper Algorithm 1:
- Standardize X, y
- Eigen-decompose XX^T
- Iterate to select ridge tuning parameter r via Newton's method
- Return final Ridge-HOLP coefficients and feature ranks (1 = most relevant)

Paper defaults (empirical): r0=10, m_tilde=floor(n/log(n)), c=1000, delta=0.01, qmax=10

"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class AirHOLPResult:
    beta_r: np.ndarray          # (p,)
    rank_r: np.ndarray          # (p,)  1=best
    beta_r0: np.ndarray         # (p,)  ridge-holp at initial r0 (first iter)
    rank_r0: np.ndarray         # (p,)
    r: float                    # selected ridge parameter
    iters_used: int


def _standardize_xy(X: np.ndarray, y: np.ndarray):
    """
    Match R code:
      X <- X - colMeans(X); X <- X / sqrt(colMeans(X^2))
      y <- (y - mean(y)) / sd(y)
    Note: R's sd() uses n-1 (ddof=1).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n, p = X.shape

    x_mean = X.mean(axis=0)
    Xc = X - x_mean

    # sqrt(colMeans(X^2)) after centering => sqrt(mean((X-mean)^2)) with ddof=0
    x_scale = np.sqrt(np.mean(Xc ** 2, axis=0))
    x_scale = np.where(x_scale == 0, 1.0, x_scale)
    Xs = Xc / x_scale

    y_mean = y.mean()
    y_sd = y.std(ddof=1)
    if y_sd == 0:
        y_sd = 1.0
    ys = (y - y_mean) / y_sd

    return Xs, ys, x_mean, x_scale, y_mean, y_sd


def _rank_desc_abs(beta: np.ndarray, rng: np.random.Generator):
    """
    Return ranks where 1 = largest |beta|.
    R code uses rank(-abs(beta), ties.method="random").
    We emulate random tie-breaking by adding tiny noise.
    """
    scores = np.abs(beta)
    noise = rng.uniform(low=-1e-12, high=1e-12, size=scores.shape)
    order = np.argsort(-(scores + noise))  # descending
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1..p
    return ranks, order


def _newton_root(Z1, Z2, x0: float, tol: float = 1e-3, m: int = 30):
    """
    Simple Newton solver for Z1(x)=0 with derivative Z2(x).
    Mirrors cmna::newton usage in R reference.
    """
    x = float(x0)
    for _ in range(m):
        f = float(Z1(x))
        fp = float(Z2(x))
        if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1e-18:
            break
        step = f / fp
        x_new = x - step
        if x_new <= 0:
            x_new = max(1e-12, x / 2.0)
        if abs(x_new - x) <= tol * max(1.0, abs(x)):
            x = x_new
            break
        x = x_new
    return float(x)


def air_holp(
    X: np.ndarray,
    y: np.ndarray,
    threshold: int | None = None,
    r0: float = 10.0,
    adapt: bool = True,
    iter_max: int = 10,
    c: float = 1000.0,
    delta: float = 0.01,
    random_state: int = 0,
):
    """
    Parameters follow the R function:
      AirHOLP(X, y, Threshold, r0=10, adapt=TRUE, iter=10, ...)
    For paper-faithful defaults:
      threshold = floor(n/log(n)) (if None)
      iter_max = 10
      r confined to [1e-4, c*sqrt(n)]  (c=1000)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if threshold is None:
        threshold = max(1, int(math.floor(n / max(1.0, math.log(n)))))
    threshold = int(min(max(1, threshold), p))

    Xs, ys, *_ = _standardize_xy(X, y)

    # Eigen decomposition of XX^T
    XXT = Xs @ Xs.T
    # symmetric -> eigh
    lam, U = np.linalg.eigh(XXT)
    # numerical safety: clip small negatives
    lam = np.clip(lam, 0.0, None)

    XU = Xs.T @ U
    UTy = U.T @ ys

    # Precompute terms used by Z functions in the R code
    yUD2UTy = (UTy ** 2) * (lam ** 2)

    # penalty bounds
    r_min = 1e-4
    r_max = c * math.sqrt(n)

    # init
    r = float(r0)
    r_prev = float(r0)
    beta_r0 = None
    rank_r0 = None
    it_used = 1

    if not adapt:
        # Just Ridge-HOLP with fixed r0
        beta = XU @ ((1.0 / (lam + r)) * UTy)
        rank, _ = _rank_desc_abs(beta, rng)
        return AirHOLPResult(
            beta_r=beta, rank_r=rank,
            beta_r0=beta, rank_r0=rank,
            r=r, iters_used=1
        )

    for j in range(1, iter_max + 1):
        # Initial screening with current r
        beta_temp = XU @ ((1.0 / (lam + r)) * UTy)
        rank_temp, order = _rank_desc_abs(beta_temp, rng)
        Xs_screen = Xs[:, order[:threshold]]

        if j == 1:
            beta_r0 = beta_temp.copy()
            rank_r0 = rank_temp.copy()

        # OLS on screened variables to estimate expected response y0 (paper suggests OLS)
        XtX = Xs_screen.T @ Xs_screen
        XtX = XtX + (1e-12 * np.eye(threshold))
        coef_ols = np.linalg.solve(XtX, Xs_screen.T @ ys)
        y_hat = Xs_screen @ coef_ols  # ys in R code is "ys"

        # ysUDUTy in R:
        # ysUDUTy_k = (y_hat^T U)_k * lam_k * UTy_k
        yhatTU = (y_hat.reshape(1, -1) @ U).reshape(-1)  # (n,)
        ysUDUTy = (yhatTU * lam) * UTy

        def Z(lam_r):
            a = 1.0 / (lam + lam_r)
            return (a ** 2) @ yUD2UTy - 2.0 * (a @ ysUDUTy)

        def Z1(lam_r):
            a = 1.0 / (lam + lam_r)
            return (-2.0 * (a ** 3) @ yUD2UTy) + (2.0 * (a ** 2) @ ysUDUTy)

        def Z2(lam_r):
            a = 1.0 / (lam + lam_r)
            return (6.0 * (a ** 4) @ yUD2UTy) - (4.0 * (a ** 3) @ ysUDUTy)

        # Newton solve for root of Z1 = 0 (R starts at 0.0001)
        r_new = _newton_root(Z1, Z2, x0=r_min, tol=1e-3, m=30)

        # clamp + boundary checks (match R code intent)
        r_new = float(np.clip(r_new, r_min, r_max))
        if Z(r_max) < Z(r_new):
            r_new = r_max
        if Z(r_min) < Z(r_new):
            r_new = r_min

        # update and stop condition |r_{i+1}-r_i| < delta*r_{i+1}  (paper Algorithm 1)
        it_used = j
        if abs(r_new - r_prev) < delta * max(r_new, 1e-12):
            r = r_new
            break

        r_prev = r
        r = r_new

    beta_final = XU @ ((1.0 / (lam + r)) * UTy)
    rank_final, _ = _rank_desc_abs(beta_final, rng)

    return AirHOLPResult(
        beta_r=beta_final,
        rank_r=rank_final,
        beta_r0=beta_r0,
        rank_r0=rank_r0,
        r=r,
        iters_used=it_used,
    )
