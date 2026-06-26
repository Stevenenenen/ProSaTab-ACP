# -*- coding: utf-8 -*-
"""Air-HOLP feature ranking module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AirHOLPResult:
    beta_r: np.ndarray
    rank_r: np.ndarray
    beta_r0: np.ndarray
    rank_r0: np.ndarray
    r: float
    iters_used: int


def _standardize_xy(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_mean = X.mean(axis=0)
    X_centered = X - x_mean

    x_scale = np.sqrt(np.mean(X_centered ** 2, axis=0))
    x_scale = np.where(x_scale == 0, 1.0, x_scale)
    X_scaled = X_centered / x_scale

    y_mean = y.mean()
    y_scale = y.std(ddof=1)
    if y_scale == 0:
        y_scale = 1.0
    y_scaled = (y - y_mean) / y_scale

    return X_scaled, y_scaled


def _rank_desc_abs(beta: np.ndarray, rng: np.random.Generator):
    scores = np.abs(beta)
    noise = rng.uniform(low=-1e-12, high=1e-12, size=scores.shape)
    order = np.argsort(-(scores + noise))

    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)

    return ranks, order


def _newton_root(Z1, Z2, x0: float, tol: float = 1e-3, max_iter: int = 30):
    x = float(x0)

    for _ in range(max_iter):
        f = float(Z1(x))
        fp = float(Z2(x))

        if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1e-18:
            break

        x_new = x - f / fp

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
) -> AirHOLPResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if threshold is None:
        threshold = max(1, int(math.floor(n / max(1.0, math.log(n)))))
    threshold = int(min(max(1, threshold), p))

    Xs, ys = _standardize_xy(X, y)

    XXT = Xs @ Xs.T
    lam, U = np.linalg.eigh(XXT)
    lam = np.clip(lam, 0.0, None)

    XU = Xs.T @ U
    UTy = U.T @ ys
    yUD2UTy = (UTy ** 2) * (lam ** 2)

    r_min = 1e-4
    r_max = c * math.sqrt(n)
    r = float(r0)
    r_prev = float(r0)

    beta_r0 = None
    rank_r0 = None
    iters_used = 1

    if not adapt:
        beta = XU @ ((1.0 / (lam + r)) * UTy)
        rank, _ = _rank_desc_abs(beta, rng)
        return AirHOLPResult(
            beta_r=beta,
            rank_r=rank,
            beta_r0=beta,
            rank_r0=rank,
            r=r,
            iters_used=1,
        )

    for j in range(1, iter_max + 1):
        beta_temp = XU @ ((1.0 / (lam + r)) * UTy)
        rank_temp, order = _rank_desc_abs(beta_temp, rng)
        X_screen = Xs[:, order[:threshold]]

        if j == 1:
            beta_r0 = beta_temp.copy()
            rank_r0 = rank_temp.copy()

        XtX = X_screen.T @ X_screen
        XtX += 1e-12 * np.eye(threshold)
        coef_ols = np.linalg.solve(XtX, X_screen.T @ ys)
        y_hat = X_screen @ coef_ols

        yhatTU = (y_hat.reshape(1, -1) @ U).reshape(-1)
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

        r_new = _newton_root(Z1, Z2, x0=r_min, tol=1e-3, max_iter=30)
        r_new = float(np.clip(r_new, r_min, r_max))

        if Z(r_max) < Z(r_new):
            r_new = r_max
        if Z(r_min) < Z(r_new):
            r_new = r_min

        iters_used = j

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
        iters_used=iters_used,
    )
