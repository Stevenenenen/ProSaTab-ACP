# -*- coding: utf-8 -*-
"""Stable Lasso feature selector with Air-HOLP ranking."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Lasso

from air_holp import air_holp


def _standardize_X(
    X: np.ndarray,
    mean_: Optional[np.ndarray] = None,
    scale_: Optional[np.ndarray] = None,
):
    X = np.asarray(X, dtype=np.float64)

    if mean_ is None:
        mean_ = X.mean(axis=0)

    X_centered = X - mean_

    if scale_ is None:
        scale_ = np.sqrt(np.mean(X_centered ** 2, axis=0))
        scale_ = np.where(scale_ == 0, 1.0, scale_)

    return X_centered / scale_, mean_, scale_


def _standardize_y(y: np.ndarray):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_mean = y.mean()
    y_scale = y.std(ddof=1)

    if y_scale == 0:
        y_scale = 1.0

    return (y - y_mean) / y_scale, y_mean, y_scale


def _nogueira_stability(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=np.float64)
    _, p = M.shape

    if p <= 1:
        return 1.0

    q = float(M.mean())

    if q <= 1e-12 or (1.0 - q) <= 1e-12:
        return 1.0

    vbar = float(M.var(axis=0, ddof=0).mean())
    stability = 1.0 - (p / (p - 1.0)) * (vbar / (q * (1.0 - q)))

    return float(np.clip(stability, -1.0, 1.0))


def _fit_weighted_lasso_squared_loss(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    w: np.ndarray,
    eps: float = 1e-6,
    max_iter: int = 5000,
    tol: float = 1e-4,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    w_eff = np.maximum(w, eps)
    X_scaled = X / w_eff

    n = X.shape[0]
    alpha = float(lam) / (2.0 * n)

    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
        random_state=0,
    )
    model.fit(X_scaled, y)

    theta = model.coef_
    beta = theta / w_eff

    return beta


def _make_lambda_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_lambdas: int = 50,
    lam_ratio: float = 1e-3,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    lam_max = 2.0 * np.max(np.abs(X.T @ y))
    lam_max = float(max(lam_max, 1e-12))
    lam_min = lam_max * float(lam_ratio)

    return np.exp(np.linspace(math.log(lam_max), math.log(lam_min), int(n_lambdas)))


def _subsample_indices(n: int, frac: float, rng: np.random.Generator):
    m = max(2, int(math.floor(n * frac)))
    return rng.choice(n, size=m, replace=False)


@dataclass
class _SelectorState:
    selected_idx: list
    x_mean: list
    x_scale: list
    weights: list
    ranks: list
    airholp_r: float
    airholp_iters: int
    threshold: int
    pi_thr: float
    subsample_frac: float
    B: int
    lambda_grid: list
    lambda_chosen: float
    stability_grid: list


class StableLassoSelector:
    def __init__(
        self,
        target_k: Optional[int] = None,
        standardize: bool = True,
        random_state: int = 42,
        verbose: bool = False,
        pi_thr: float = 0.8,
        B: int = 100,
        subsample_frac: float = 0.5,
        n_lambdas: int = 50,
        lam_ratio: float = 1e-3,
        threshold: Optional[int] = None,
        r0: float = 10.0,
        c: float = 1000.0,
        delta: float = 0.01,
        iter_max: int = 10,
        weight_eps: float = 1e-6,
        lasso_max_iter: int = 5000,
        lasso_tol: float = 1e-4,
    ):
        self.target_k = target_k
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose

        self.pi_thr = pi_thr
        self.B = B
        self.subsample_frac = subsample_frac
        self.n_lambdas = n_lambdas
        self.lam_ratio = lam_ratio

        self.threshold = threshold
        self.r0 = r0
        self.c = c
        self.delta = delta
        self.iter_max = iter_max
        self.weight_eps = float(weight_eps)
        self.lasso_max_iter = int(lasso_max_iter)
        self.lasso_tol = float(lasso_tol)

        self.selected_idx_: Optional[np.ndarray] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.x_scale_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.ranks_: Optional[np.ndarray] = None
        self.airholp_r_: Optional[float] = None
        self.airholp_iters_: Optional[int] = None
        self.lambda_grid_: Optional[np.ndarray] = None
        self.stability_grid_: Optional[np.ndarray] = None
        self.lambda_chosen_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.standardize:
            Xs, self.x_mean_, self.x_scale_ = _standardize_X(X)
        else:
            Xs = X.copy()
            self.x_mean_ = np.zeros(p)
            self.x_scale_ = np.ones(p)

        ys, _, _ = _standardize_y(y)

        threshold = self.threshold
        if threshold is None:
            threshold = max(1, int(math.floor(n / max(1.0, math.log(n)))))
        threshold = int(min(max(1, threshold), p))
        self.threshold = threshold

        ah = air_holp(
            Xs,
            ys,
            threshold=threshold,
            r0=self.r0,
            adapt=True,
            iter_max=self.iter_max,
            c=self.c,
            delta=self.delta,
            random_state=self.random_state,
        )

        self.ranks_ = ah.rank_r.astype(np.int64)
        self.airholp_r_ = float(ah.r)
        self.airholp_iters_ = int(ah.iters_used)

        ranks = self.ranks_.astype(np.float64)
        self.weights_ = 1.0 - (1.0 / np.maximum(ranks, 1.0))

        self.lambda_grid_ = _make_lambda_grid(
            Xs,
            ys,
            n_lambdas=self.n_lambdas,
            lam_ratio=self.lam_ratio,
        )

        stability_values = []

        for lam in self.lambda_grid_:
            M = np.zeros((self.B, p), dtype=np.int8)

            for b in range(self.B):
                idx = _subsample_indices(n, self.subsample_frac, rng)
                beta = _fit_weighted_lasso_squared_loss(
                    Xs[idx],
                    ys[idx],
                    lam=lam,
                    w=self.weights_,
                    eps=self.weight_eps,
                    max_iter=self.lasso_max_iter,
                    tol=self.lasso_tol,
                )
                M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)

            stability_values.append(_nogueira_stability(M))

        self.stability_grid_ = np.asarray(stability_values, dtype=np.float64)

        target_stability = 0.75
        idx_ge = np.where(self.stability_grid_ >= target_stability)[0]

        if len(idx_ge) > 0:
            chosen_i = int(idx_ge[-1])
        else:
            chosen_i = self._choose_lambda_by_one_sd(Xs, ys, n, p)

        self.lambda_chosen_ = float(self.lambda_grid_[chosen_i])

        selection_prob = self._selection_probability(Xs, ys, n, p)
        selected = np.where(selection_prob >= self.pi_thr)[0]

        if selected.size == 0:
            order = np.lexsort((self.ranks_, -selection_prob))
            selected = order[:1]

        if self.target_k is not None:
            selected = self._force_target_k(selected, selection_prob)

        self.selected_idx_ = selected.astype(np.int64)

        if self.verbose:
            print(
                f"[StableLassoSelector] n={n} p={p} threshold={threshold} "
                f"airholp_r={self.airholp_r_:.4g} lambda={self.lambda_chosen_:.4g} "
                f"K={len(self.selected_idx_)}"
            )

        return self

    def _choose_lambda_by_one_sd(self, Xs: np.ndarray, ys: np.ndarray, n: int, p: int) -> int:
        G = min(10, self.B)
        chunk = max(1, self.B // G)
        stability_chunks = np.zeros((len(self.lambda_grid_), G), dtype=np.float64)
        rng = np.random.default_rng(self.random_state + 999)

        for gi in range(G):
            for li, lam in enumerate(self.lambda_grid_):
                M = np.zeros((chunk, p), dtype=np.int8)

                for b in range(chunk):
                    idx = _subsample_indices(n, self.subsample_frac, rng)
                    beta = _fit_weighted_lasso_squared_loss(
                        Xs[idx],
                        ys[idx],
                        lam=lam,
                        w=self.weights_,
                        eps=self.weight_eps,
                        max_iter=self.lasso_max_iter,
                        tol=self.lasso_tol,
                    )
                    M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)

                stability_chunks[li, gi] = _nogueira_stability(M)

        stability_mean = stability_chunks.mean(axis=1)
        stability_sd = stability_chunks.std(axis=1, ddof=1) if G > 1 else np.zeros_like(stability_mean)

        best_i = int(np.argmax(stability_mean))
        threshold = stability_mean[best_i] - stability_sd[best_i]
        candidates = np.where(stability_mean >= threshold)[0]

        return int(candidates[-1]) if len(candidates) > 0 else best_i

    def _selection_probability(self, Xs: np.ndarray, ys: np.ndarray, n: int, p: int):
        M = np.zeros((self.B, p), dtype=np.int8)
        rng = np.random.default_rng(self.random_state + 2025)

        for b in range(self.B):
            idx = _subsample_indices(n, self.subsample_frac, rng)
            beta = _fit_weighted_lasso_squared_loss(
                Xs[idx],
                ys[idx],
                lam=self.lambda_chosen_,
                w=self.weights_,
                eps=self.weight_eps,
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol,
            )
            M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)

        return M.mean(axis=0)

    def _force_target_k(self, selected: np.ndarray, selection_prob: np.ndarray):
        k = int(self.target_k)
        order = np.lexsort((self.ranks_, -selection_prob))
        chosen = []
        chosen_set = set()

        for j in order:
            if j in selected and int(j) not in chosen_set:
                chosen.append(int(j))
                chosen_set.add(int(j))
            if len(chosen) >= k:
                break

        if len(chosen) < k:
            for j in order:
                if int(j) not in chosen_set:
                    chosen.append(int(j))
                    chosen_set.add(int(j))
                if len(chosen) >= k:
                    break

        return np.asarray(chosen[:k], dtype=np.int64)

    def transform(self, X: np.ndarray):
        if self.selected_idx_ is None:
            raise RuntimeError("StableLassoSelector is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)

        if self.standardize:
            Xs, _, _ = _standardize_X(X, mean_=self.x_mean_, scale_=self.x_scale_)
        else:
            Xs = X

        return Xs[:, self.selected_idx_].astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        if self.selected_idx_ is None:
            raise RuntimeError("Nothing to save: selector is not fitted.")

        state = _SelectorState(
            selected_idx=self.selected_idx_.tolist(),
            x_mean=self.x_mean_.tolist(),
            x_scale=self.x_scale_.tolist(),
            weights=self.weights_.tolist(),
            ranks=self.ranks_.tolist(),
            airholp_r=float(self.airholp_r_),
            airholp_iters=int(self.airholp_iters_),
            threshold=int(self.threshold),
            pi_thr=float(self.pi_thr),
            subsample_frac=float(self.subsample_frac),
            B=int(self.B),
            lambda_grid=self.lambda_grid_.tolist(),
            lambda_chosen=float(self.lambda_chosen_),
            stability_grid=self.stability_grid_.tolist(),
        )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, standardize: bool = True):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        obj = cls(
            target_k=len(data["selected_idx"]) if data.get("selected_idx") else None,
            standardize=standardize,
            random_state=0,
            verbose=False,
            pi_thr=float(data["pi_thr"]),
            B=int(data["B"]),
            subsample_frac=float(data["subsample_frac"]),
            n_lambdas=len(data["lambda_grid"]),
            threshold=int(data["threshold"]),
        )

        obj.selected_idx_ = np.asarray(data["selected_idx"], dtype=np.int64)
        obj.x_mean_ = np.asarray(data["x_mean"], dtype=np.float64)
        obj.x_scale_ = np.asarray(data["x_scale"], dtype=np.float64)
        obj.weights_ = np.asarray(data["weights"], dtype=np.float64)
        obj.ranks_ = np.asarray(data["ranks"], dtype=np.int64)
        obj.airholp_r_ = float(data["airholp_r"])
        obj.airholp_iters_ = int(data["airholp_iters"])
        obj.lambda_grid_ = np.asarray(data["lambda_grid"], dtype=np.float64)
        obj.lambda_chosen_ = float(data["lambda_chosen"])
        obj.stability_grid_ = np.asarray(data["stability_grid"], dtype=np.float64)

        return obj
