# stable_lasso_selector.py
# -*- coding: utf-8 -*-
"""
Stable Lasso selector (paper-faithful route):
1) Air-HOLP ranking on training data
   - screening threshold d = floor(n/log(n))
2) weights w_j = 1 - 1/r_j
3) Solve Stable Lasso: ||Y - Xb||^2 + λ ||w ⊙ b||_1
   - Implement via feature scaling trick (penalty factors)
4) Embed in Stability Selection:
   - weights fixed across all sub-samples
   - tune λ via stability: λ_stable is smallest λ with stability >= 0.75,
     else λ_stable-1sd
5) Select variables by selection probability >= pi_thr (optionally force target_k dims)

Note: Paper is formulated for squared-loss regression.
For binary labels, we still use squared-loss Lasso to do feature selection (same as many screening pipelines).
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import Lasso

from air_holp import air_holp


def _standardize_X(X: np.ndarray, mean_: Optional[np.ndarray] = None, scale_: Optional[np.ndarray] = None):
    X = np.asarray(X, dtype=np.float64)
    if mean_ is None:
        mean_ = X.mean(axis=0)
    Xc = X - mean_
    if scale_ is None:
        # match Air-HOLP code style: sqrt(mean(Xc^2)) (ddof=0)
        scale_ = np.sqrt(np.mean(Xc ** 2, axis=0))
        scale_ = np.where(scale_ == 0, 1.0, scale_)
    return Xc / scale_, mean_, scale_


def _standardize_y(y: np.ndarray):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = y.mean()
    sd = y.std(ddof=1)
    if sd == 0:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def _nogueira_stability(M: np.ndarray):
    """
    Nogueira et al. (2018) stability:
    A scaled average of column-wise variances (paper wording).
    Common closed form:
      q = mean(M)
      vbar = mean_j Var(M[:,j])
      stability = 1 - (p/(p-1)) * vbar / (q*(1-q))
    Handle edge cases q in {0,1}.
    """
    M = np.asarray(M, dtype=np.float64)
    B, p = M.shape
    if p <= 1:
        return 1.0
    q = float(M.mean())
    if q <= 1e-12 or (1.0 - q) <= 1e-12:
        return 1.0
    vbar = float(M.var(axis=0, ddof=0).mean())
    stab = 1.0 - (p / (p - 1.0)) * (vbar / (q * (1.0 - q)))
    # numerical clip
    return float(np.clip(stab, -1.0, 1.0))


def _fit_weighted_lasso_squared_loss(X: np.ndarray, y: np.ndarray, lam: float, w: np.ndarray, eps: float = 1e-6):
    """
    Solve: ||y - Xb||^2 + lam * ||w ⊙ b||_1
    Using substitution θ = w ⊙ b, X' = X / w:
      ||y - X'(θ)||^2 + lam * ||θ||_1
    Then b = θ / w.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)

    w_eff = np.maximum(w, eps)
    Xs = X / w_eff  # columnwise

    n = X.shape[0]
    # sklearn Lasso objective: (1/(2n))||y - Xθ||^2 + alpha||θ||_1
    alpha = float(lam) / (2.0 * n)

    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=5000,
        tol=1e-4,
        selection="cyclic",
        random_state=0,
    )
    model.fit(Xs, y)
    theta = model.coef_
    beta = theta / w_eff
    return beta


def _make_lambda_grid(X: np.ndarray, y: np.ndarray, n_lambdas: int = 50, lam_ratio: float = 1e-3):
    """
    A practical grid (paper says CV used to form Λ; we do a standard lasso-style grid).
    For standardized X,y: lam_max ~= 2 * max |X^T y| (depending on definition).
    We'll build logspace [lam_max, lam_max*lam_ratio].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = X.shape[0]
    # For objective ||y-Xb||^2 + lam||b||_1, a safe lam_max scale:
    lam_max = 2.0 * np.max(np.abs(X.T @ y))
    lam_max = float(max(lam_max, 1e-12))
    lam_min = lam_max * float(lam_ratio)
    grid = np.exp(np.linspace(math.log(lam_max), math.log(lam_min), int(n_lambdas)))
    return grid


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
    """
    Drop-in replacement for your current selector.

    Parameters:
      target_k: if not None, FORCE output dimension = target_k
                (not strictly in paper, but convenient for TabPFN and matches你现在K=1024的需求)
      pi_thr: selection prob threshold
      B: #subsamples for stability selection
      subsample_frac: fraction per subsample (0.5 is typical)
      n_lambdas: grid size
    """
    def __init__(
        self,
        target_k: Optional[int] = None,
        standardize: bool = True,
        random_state: int = 42,
        verbose: bool = False,
        # paper-ish knobs:
        pi_thr: float = 0.8,
        B: int = 100,
        subsample_frac: float = 0.5,
        n_lambdas: int = 50,
        lam_ratio: float = 1e-3,
        # Air-HOLP knobs:
        threshold: Optional[int] = None,   # default floor(n/log(n))
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

        # fitted
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
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        # Standardize X (and y for ranking/selection)
        if self.standardize:
            Xs, self.x_mean_, self.x_scale_ = _standardize_X(X)
        else:
            Xs, self.x_mean_, self.x_scale_ = X.copy(), np.zeros(p), np.ones(p)

        ys, _, _ = _standardize_y(y)

        # 1) Air-HOLP ranking
        # threshold default floor(n/log(n)) as in Stable Lasso paper
        thr = self.threshold
        if thr is None:
            thr = max(1, int(math.floor(n / max(1.0, math.log(n)))))
        thr = int(min(max(1, thr), p))
        self.threshold = thr

        ah = air_holp(
            Xs, ys,
            threshold=thr,
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

        # 2) weights w_j = 1 - 1/r_j
        # NOTE: our ranks_ is 1=best (consistent with the weight shape: top gets smallest penalty).
        ranks = self.ranks_.astype(np.float64)
        self.weights_ = 1.0 - (1.0 / np.maximum(ranks, 1.0))

        # 3) Build λ grid
        self.lambda_grid_ = _make_lambda_grid(Xs, ys, n_lambdas=self.n_lambdas, lam_ratio=self.lam_ratio)

        # 4) Stability selection over λ grid (weights fixed across subsamples)
        stab_vals = []
        prob_vals = []  # selection probabilities for each λ (optional debug)
        for lam in self.lambda_grid_:
            M = np.zeros((self.B, p), dtype=np.int8)
            for b in range(self.B):
                idx = _subsample_indices(n, self.subsample_frac, rng)
                beta = _fit_weighted_lasso_squared_loss(Xs[idx], ys[idx], lam=lam, w=self.weights_)
                M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)
            stab = _nogueira_stability(M)
            stab_vals.append(stab)
            prob_vals.append(M.mean(axis=0))

        self.stability_grid_ = np.asarray(stab_vals, dtype=np.float64)

        # 5) Choose λ: λ_stable (>=0.75) else λ_stable-1sd
        # We approximate "1sd" by splitting B subsamples into 10 chunks and computing stability per chunk.
        target_stab = 0.75
        idx_ge = np.where(self.stability_grid_ >= target_stab)[0]
        if len(idx_ge) > 0:
            chosen_i = int(idx_ge[0])  # smallest λ in grid order (grid is descending -> careful)
            # Our grid goes from lam_max -> lam_min, i.e. decreasing.
            # "smallest λ achieving >=0.75" means lam_min among those => last index among >= target.
            chosen_i = int(idx_ge[-1])
        else:
            # estimate SD around each stability by chunking
            G = 10
            G = min(G, self.B)
            chunk = self.B // G
            stab_chunk = np.zeros((len(self.lambda_grid_), G), dtype=np.float64)
            rng2 = np.random.default_rng(self.random_state + 999)

            for gi in range(G):
                # re-run smaller stability selection to approximate variance
                # (cheap-ish; still heavy but acceptable for faithful tuning)
                for li, lam in enumerate(self.lambda_grid_):
                    M = np.zeros((chunk, p), dtype=np.int8)
                    for b in range(chunk):
                        idx = _subsample_indices(n, self.subsample_frac, rng2)
                        beta = _fit_weighted_lasso_squared_loss(Xs[idx], ys[idx], lam=lam, w=self.weights_)
                        M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)
                    stab_chunk[li, gi] = _nogueira_stability(M)

            stab_mean = stab_chunk.mean(axis=1)
            stab_sd = stab_chunk.std(axis=1, ddof=1) if G > 1 else np.zeros_like(stab_mean)

            best_i = int(np.argmax(stab_mean))
            best = stab_mean[best_i]
            thr_1sd = best - stab_sd[best_i]

            # choose "smallest λ whose stability within 1sd of max" => again smallest λ => lam_min side
            ok = np.where(stab_mean >= thr_1sd)[0]
            chosen_i = int(ok[-1]) if len(ok) > 0 else best_i

        self.lambda_chosen_ = float(self.lambda_grid_[chosen_i])

        # 6) Recompute selection probabilities at chosen λ, then select by pi_thr
        # (This is the selection set you will use to transform features.)
        lam = self.lambda_chosen_
        M = np.zeros((self.B, p), dtype=np.int8)
        rng3 = np.random.default_rng(self.random_state + 2025)
        for b in range(self.B):
            idx = _subsample_indices(n, self.subsample_frac, rng3)
            beta = _fit_weighted_lasso_squared_loss(Xs[idx], ys[idx], lam=lam, w=self.weights_)
            M[b, :] = (np.abs(beta) > 1e-12).astype(np.int8)
        sel_prob = M.mean(axis=0)

        selected = np.where(sel_prob >= self.pi_thr)[0]
        if selected.size == 0:
            # fallback: take top by sel_prob then Air-HOLP rank
            order = np.lexsort((self.ranks_, -sel_prob))
            selected = order[:1]

        # Optional: force fixed K (useful for your TabPFN pipeline)
        if self.target_k is not None:
            k = int(self.target_k)
            order = np.lexsort((self.ranks_, -sel_prob))  # prob desc, then rank asc
            chosen = []
            chosen_set = set()

            # first keep those above threshold in good order
            for j in order:
                if j in selected and j not in chosen_set:
                    chosen.append(int(j))
                    chosen_set.add(int(j))
                if len(chosen) >= k:
                    break

            # then fill up if not enough
            if len(chosen) < k:
                for j in order:
                    if int(j) not in chosen_set:
                        chosen.append(int(j))
                        chosen_set.add(int(j))
                    if len(chosen) >= k:
                        break

            selected = np.asarray(chosen[:k], dtype=np.int64)

        self.selected_idx_ = selected.astype(np.int64)

        if self.verbose:
            print(
                f"[StableLassoSelector] n={n} p={p} thr={thr} "
                f"AirHOLP(r={self.airholp_r_:.4g}, it={self.airholp_iters_}) "
                f"lambda={self.lambda_chosen_:.4g} pi_thr={self.pi_thr} "
                f"K={len(self.selected_idx_)}"
            )

        return self

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
            raise RuntimeError("Nothing to save: selector not fitted.")
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
            d: Dict[str, Any] = json.load(f)

        obj = cls(
            target_k=len(d["selected_idx"]) if d.get("selected_idx") else None,
            standardize=standardize,
            random_state=0,
            verbose=False,
            pi_thr=float(d["pi_thr"]),
            B=int(d["B"]),
            subsample_frac=float(d["subsample_frac"]),
            n_lambdas=len(d["lambda_grid"]),
            threshold=int(d["threshold"]),
        )
        obj.selected_idx_ = np.asarray(d["selected_idx"], dtype=np.int64)
        obj.x_mean_ = np.asarray(d["x_mean"], dtype=np.float64)
        obj.x_scale_ = np.asarray(d["x_scale"], dtype=np.float64)
        obj.weights_ = np.asarray(d["weights"], dtype=np.float64)
        obj.ranks_ = np.asarray(d["ranks"], dtype=np.int64)
        obj.airholp_r_ = float(d["airholp_r"])
        obj.airholp_iters_ = int(d["airholp_iters"])
        obj.lambda_grid_ = np.asarray(d["lambda_grid"], dtype=np.float64)
        obj.lambda_chosen_ = float(d["lambda_chosen"])
        obj.stability_grid_ = np.asarray(d["stability_grid"], dtype=np.float64)
        return obj
