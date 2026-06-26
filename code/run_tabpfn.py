import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.model_loading import save_fitted_tabpfn_model

from stable_lasso_selector import StableLassoSelector


# =========================================================
# 0. Reproducibility
# =========================================================
def set_global_seed(seed: int = 42) -> None:
    """Set random seeds to reduce non-determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 1. Metrics
# =========================================================
def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate commonly used ACP classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    aupr = auc(recall, precision)

    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return accuracy, auc_score, aupr, f1, mcc, sn, sp


# =========================================================
# 2. Read selected indices from StableLassoSelector
# =========================================================
def get_selected_indices_from_selector(selector) -> np.ndarray:
    """
    Read selected feature indices from a StableLassoSelector object.

    The returned indices are local indices relative to the current input feature
    matrix of the current chain step. They are not necessarily direct indices in
    the original feature space.
    """
    candidate_attrs = [
        "selected_indices_",
        "selected_idx_",
        "support_indices_",
        "selected_features_",
        "feature_indices_",
        "indices_",
        "support_",
    ]

    for attr in candidate_attrs:
        if not hasattr(selector, attr):
            continue

        value = getattr(selector, attr)

        if value is None:
            continue

        value = np.asarray(value)

        if value.dtype == bool:
            return np.where(value)[0].astype(int)

        if np.issubdtype(value.dtype, np.integer):
            return value.astype(int)

    print("\n[ERROR] Cannot find selected feature indices in selector.")
    print("[DEBUG] selector.__dict__.keys():")
    print(list(selector.__dict__.keys()))

    raise AttributeError(
        "Cannot find selected feature indices in StableLassoSelector. "
        "Please check stable_lasso_selector.py and add the correct attribute "
        "name to candidate_attrs, such as selected_indices_ or support_."
    )


# =========================================================
# 3. One-time feature selection outside folds + cache
# =========================================================
def fit_selector_and_cache(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_sets: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
    tag: str,
    target_k: int,
    random_state: int = 42,
    verbose: bool = False,
    threshold=None,
    r0=10.0,
    c=1000.0,
    delta=0.01,
    iter_max=10,
    B=100,
    subsample_frac=0.5,
    pi_thr=0.8,
    n_lambdas=50,
    lam_ratio=1e-3,
    force_rerun_selector: bool = False,
):
    """Fit StableLassoSelector for a target K and cache transformed features."""
    os.makedirs(out_dir, exist_ok=True)

    selector_path = os.path.join(out_dir, f"selector_{tag}_K{target_k}.json")
    X_train_path = os.path.join(out_dir, f"X_train_{tag}_K{target_k}.npy")
    y_train_path = os.path.join(out_dir, f"y_train_{tag}.npy")
    selected_local_indices_path = os.path.join(
        out_dir,
        f"selected_local_indices_{tag}_K{target_k}.npy",
    )

    if (
        (not force_rerun_selector)
        and os.path.exists(selector_path)
        and os.path.exists(X_train_path)
        and os.path.exists(y_train_path)
        and os.path.exists(selected_local_indices_path)
    ):
        Xk = np.load(X_train_path)
        yk = np.load(y_train_path)
        selected_local_indices = np.load(selected_local_indices_path).astype(int)

        X_tests_k = []
        ok = True

        for i in range(len(test_sets)):
            xte_path = os.path.join(out_dir, f"X_test{i + 1}_{tag}_K{target_k}.npy")
            yte_path = os.path.join(out_dir, f"y_test{i + 1}_{tag}.npy")

            if os.path.exists(xte_path) and os.path.exists(yte_path):
                X_tests_k.append((np.load(xte_path), np.load(yte_path)))
            else:
                ok = False
                break

        if ok:
            print(f"[CACHE HIT - SELECTOR] K={target_k}")
            print(f"[CACHE HIT - LOCAL INDEX] -> {selected_local_indices_path}")

            return (
                selector_path,
                Xk.astype(np.float32),
                yk.astype(int),
                [(xt.astype(np.float32), yt.astype(int)) for xt, yt in X_tests_k],
                selected_local_indices,
            )

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int)

    selector = StableLassoSelector(
        target_k=int(target_k),
        standardize=True,
        random_state=int(random_state),
        verbose=bool(verbose),
        pi_thr=float(pi_thr),
        B=int(B),
        subsample_frac=float(subsample_frac),
        n_lambdas=int(n_lambdas),
        lam_ratio=float(lam_ratio),
        threshold=threshold,
        r0=float(r0),
        c=float(c),
        delta=float(delta),
        iter_max=int(iter_max),
    )

    t0 = time.perf_counter()
    Xk_train = selector.fit_transform(X_train, y_train)

    selected_local_indices = get_selected_indices_from_selector(selector)

    if len(selected_local_indices) != Xk_train.shape[1]:
        raise ValueError(
            "The number of selected indices does not match the transformed "
            f"feature dimension: len(selected_local_indices)={len(selected_local_indices)}, "
            f"Xk_train.shape[1]={Xk_train.shape[1]}"
        )

    selector.save(selector_path)
    np.save(selected_local_indices_path, selected_local_indices.astype(int))
    np.save(X_train_path, Xk_train.astype(np.float32))
    np.save(y_train_path, y_train.astype(int))

    X_tests_k = []

    for i, (X_te, y_te) in enumerate(test_sets, 1):
        X_te = np.asarray(X_te, dtype=np.float32)
        y_te = np.asarray(y_te).astype(int)

        Xk_te = selector.transform(X_te)

        xte_path = os.path.join(out_dir, f"X_test{i}_{tag}_K{target_k}.npy")
        yte_path = os.path.join(out_dir, f"y_test{i}_{tag}.npy")

        np.save(xte_path, Xk_te.astype(np.float32))
        np.save(yte_path, y_te.astype(int))

        X_tests_k.append((Xk_te.astype(np.float32), y_te.astype(int)))

    t1 = time.perf_counter()

    print(f"[SELECT+CACHE] K={target_k}, dim={Xk_train.shape[1]}, time={t1 - t0:.1f}s")
    print(f"[SELECTOR SAVED] -> {selector_path}")
    print(f"[LOCAL INDEX SAVED] -> {selected_local_indices_path}")

    return (
        selector_path,
        Xk_train.astype(np.float32),
        y_train.astype(int),
        X_tests_k,
        selected_local_indices.astype(int),
    )


# =========================================================
# 4. TabPFN 10-fold CV + independent test set evaluation
# =========================================================
def tabpfn_cv_and_test_on_fixed_features(
    X: np.ndarray,
    y: np.ndarray,
    test_sets: List[Tuple[np.ndarray, np.ndarray]],
    out_models_dir: str,
    model_tag: str,
    n_splits: int = 10,
    random_state: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Run TabPFN 10-fold CV and evaluate each fold model on test sets."""
    os.makedirs(out_models_dir, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    test_sets = [
        (np.asarray(xt, dtype=np.float32), np.asarray(yt).astype(int))
        for xt, yt in test_sets
    ]

    probs_bucket = [[] for _ in range(len(test_sets))]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_rows = []
    test_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        t0 = time.perf_counter()

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        clf = TabPFNClassifier(
            device=device,
            ignore_pretraining_limits=True,
            random_state=random_state,
        )

        clf.fit(X_tr, y_tr)

        probs_va = clf.predict_proba(X_va)[:, 1]
        pred_va = (probs_va >= 0.5).astype(int)

        acc, auc_, aupr, f1, mcc, sn, sp = calculate_metrics(y_va, pred_va, probs_va)

        cv_rows.append(
            {
                "fold": fold,
                "ACC": acc,
                "AUC": auc_,
                "AUPR": aupr,
                "F1": f1,
                "MCC": mcc,
                "Sn": sn,
                "Sp": sp,
                "K": X.shape[1],
            }
        )

        model_path = os.path.join(out_models_dir, f"{model_tag}_fold{fold}.tabpfn_fit")
        save_fitted_tabpfn_model(clf, model_path)

        for i, (X_te, y_te) in enumerate(test_sets, 1):
            probs_te = clf.predict_proba(X_te)[:, 1]
            pred_te = (probs_te >= 0.5).astype(int)

            t_acc, t_auc, t_aupr, t_f1, t_mcc, t_sn, t_sp = calculate_metrics(
                y_te,
                pred_te,
                probs_te,
            )

            test_rows.append(
                {
                    "fold": f"test_{i}_fold{fold}",
                    "ACC": t_acc,
                    "AUC": t_auc,
                    "AUPR": t_aupr,
                    "F1": t_f1,
                    "MCC": t_mcc,
                    "Sn": t_sn,
                    "Sp": t_sp,
                    "K": X.shape[1],
                }
            )

            probs_bucket[i - 1].append(probs_te)

        t1 = time.perf_counter()

        print(
            f"[TabPFN-CV] fold {fold}/{n_splits} "
            f"K={X.shape[1]} MCC={mcc:.4f} time={t1 - t0:.1f}s"
        )

    cv_df = pd.DataFrame(cv_rows)

    cv_mean = {
        "fold": "Mean",
        "ACC": f"{cv_df['ACC'].mean():.5f}±{cv_df['ACC'].std():.4f}",
        "AUC": f"{cv_df['AUC'].mean():.5f}±{cv_df['AUC'].std():.4f}",
        "AUPR": f"{cv_df['AUPR'].mean():.5f}±{cv_df['AUPR'].std():.4f}",
        "F1": f"{cv_df['F1'].mean():.5f}±{cv_df['F1'].std():.4f}",
        "MCC": f"{cv_df['MCC'].mean():.5f}±{cv_df['MCC'].std():.4f}",
        "Sn": f"{cv_df['Sn'].mean():.5f}±{cv_df['Sn'].std():.4f}",
        "Sp": f"{cv_df['Sp'].mean():.5f}±{cv_df['Sp'].std():.4f}",
        "K": f"{cv_df['K'].mean():.1f}±{cv_df['K'].std():.1f}",
    }

    cv_df = pd.concat([cv_df, pd.DataFrame([cv_mean])], ignore_index=True)

    test_df = pd.DataFrame(test_rows)

    if not test_df.empty:
        test_mean = {
            "fold": "Test_Mean",
            "ACC": f"{test_df['ACC'].mean():.5f}±{test_df['ACC'].std():.4f}",
            "AUC": f"{test_df['AUC'].mean():.5f}±{test_df['AUC'].std():.4f}",
            "AUPR": f"{test_df['AUPR'].mean():.5f}±{test_df['AUPR'].std():.4f}",
            "F1": f"{test_df['F1'].mean():.5f}±{test_df['F1'].std():.4f}",
            "MCC": f"{test_df['MCC'].mean():.5f}±{test_df['MCC'].std():.4f}",
            "Sn": f"{test_df['Sn'].mean():.5f}±{test_df['Sn'].std():.4f}",
            "Sp": f"{test_df['Sp'].mean():.5f}±{test_df['Sp'].std():.4f}",
            "K": f"{test_df['K'].mean():.1f}±{test_df['K'].std():.1f}",
        }

        test_df = pd.concat([test_df, pd.DataFrame([test_mean])], ignore_index=True)

    ens_rows = []

    for i, (X_te, y_te) in enumerate(test_sets, 1):
        if not probs_bucket[i - 1]:
            continue

        probs_ens = np.mean(np.stack(probs_bucket[i - 1], axis=0), axis=0)
        pred_ens = (probs_ens >= 0.5).astype(int)

        e_acc, e_auc, e_aupr, e_f1, e_mcc, e_sn, e_sp = calculate_metrics(
            y_te,
            pred_ens,
            probs_ens,
        )

        ens_rows.append(
            {
                "fold": f"ensemble_test_{i}",
                "ACC": e_acc,
                "AUC": e_auc,
                "AUPR": e_aupr,
                "F1": e_f1,
                "MCC": e_mcc,
                "Sn": e_sn,
                "Sp": e_sp,
                "K": X.shape[1],
            }
        )

    if ens_rows:
        test_df = pd.concat([test_df, pd.DataFrame(ens_rows)], ignore_index=True)

    return cv_df, test_df


# =========================================================
# 5. Final full-training TabPFN model per K
# =========================================================
def train_and_save_final_tabpfn(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    model_tag: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    random_state: int = 42,
):
    """Train a final TabPFN model on the full training set and save it."""
    os.makedirs(out_dir, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    clf = TabPFNClassifier(
        device=device,
        ignore_pretraining_limits=True,
        random_state=random_state,
    )

    clf.fit(X, y)

    final_model_path = os.path.join(out_dir, f"{model_tag}_FINAL.tabpfn_fit")
    save_fitted_tabpfn_model(clf, final_model_path)

    print(f"[FINAL MODEL SAVED] -> {final_model_path}")

    return final_model_path


# =========================================================
# 6. Save utilities
# =========================================================
def save_metadata(path: str, data: dict) -> None:
    """Save metadata as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[METADATA SAVED] -> {path}")


def save_df(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"[CSV SAVED] -> {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StableLasso chain feature selection and TabPFN training for ACP prediction."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default="YOUR_PROJECT_ROOT",
        help="Root directory of the project. Replace YOUR_PROJECT_ROOT with your local path.",
    )
    parser.add_argument(
        "--feature-name",
        type=str,
        default="saprot_prott5",
        help="Feature representation name.",
    )
    parser.add_argument(
        "--sampling-method-name",
        type=str,
        default="edit+mdndo",
        help="Resampling method name used in file names.",
    )
    parser.add_argument(
        "--force-rerun-selector",
        action="store_true",
        help="Regenerate selector files and selected feature caches.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser


# =========================================================
# 7. Main chain
# =========================================================
def main() -> None:
    args = build_arg_parser().parse_args()

    set_global_seed(args.seed)

    sampling_method_name = args.sampling_method_name
    feature_name = args.feature_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = Path(args.project_root)

    if str(project_root) == "YOUR_PROJECT_ROOT":
        print(
            "[WARNING] PROJECT_ROOT is still set to 'YOUR_PROJECT_ROOT'. "
            "Please replace it with your own local project path or pass --project-root."
        )

    feature_dir = project_root / "data" / "features" / feature_name / "ACP"
    resampled_dir = project_root / "data" / "resampled" / "ACP" / feature_name
    output_root = project_root / "outputs"

    print(f"Using device: {device}")
    print(f"Project root: {project_root}")
    print(f"Feature directory: {feature_dir}")
    print(f"Resampled directory: {resampled_dir}")

    # Original high-dimensional independent test features.
    test_files = [
        (
            feature_dir / "X_test_features.npy",
            feature_dir / "y_test_labels.npy",
        ),
        (
            feature_dir / "X_test_features_99.npy",
            feature_dir / "y_test_labels_99.npy",
        ),
    ]

    raw_test_sets = [(np.load(fx), np.load(fy)) for fx, fy in test_files]

    # Resampled high-dimensional training features.
    X_train_raw = np.load(
        resampled_dir / f"X_train_features_{sampling_method_name}.npy"
    )

    y_train_raw = np.load(
        resampled_dir / f"y_train_labels_{sampling_method_name}.npy"
    ).astype(int)

    print(f"[RAW TRAIN] X_train_raw.shape = {X_train_raw.shape}")
    print(f"[RAW TRAIN] y_train_raw.shape = {y_train_raw.shape}")

    for i, (xt, yt) in enumerate(raw_test_sets, 1):
        print(f"[RAW TEST {i}] X.shape = {xt.shape}, y.shape = {yt.shape}")

    # Target feature dimensions for chain selection.
    K_list = list(range(1024, 511, -20))

    if K_list[-1] != 512:
        K_list.append(512)

    root_run_dir = output_root / f"runs_chain_{feature_name}_{sampling_method_name}"
    root_run_dir.mkdir(parents=True, exist_ok=True)

    # Save label mapping.
    label_map = {
        "0": "non-ACP",
        "1": "ACP",
    }

    label_map_path = root_run_dir / "label_map.json"

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[LABEL MAP SAVED] -> {label_map_path}")

    X_curr = np.asarray(X_train_raw, dtype=np.float32)
    y_curr = np.asarray(y_train_raw, dtype=int)

    test_curr = [
        (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=int))
        for x, y in raw_test_sets
    ]

    # Map the current feature space to the original feature space.
    current_original_indices = np.arange(X_train_raw.shape[1], dtype=int)

    # Save the full selector chain from the original feature space to each target K.
    selector_chain_paths = []

    for idx, K in enumerate(K_list):
        print("\n" + "=" * 100)
        print(f"[CHAIN] Step {idx + 1}/{len(K_list)} | target K = {K}")
        print("=" * 100)

        k_dir = root_run_dir / f"K_{K}"
        cache_dir = k_dir / "cache"
        cv_models_dir = k_dir / "cv_models"
        final_model_dir = k_dir / "final_model"
        results_dir = k_dir / "results"

        for directory in [k_dir, cache_dir, cv_models_dir, final_model_dir, results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        tag = f"{feature_name}_{sampling_method_name}_chain"

        (
            selector_path,
            Xk_train,
            yk_train,
            test_k,
            selected_local_indices,
        ) = fit_selector_and_cache(
            X_train=X_curr,
            y_train=y_curr,
            test_sets=test_curr,
            out_dir=str(cache_dir),
            tag=tag,
            target_k=K,
            random_state=args.seed + idx,
            verbose=False,
            threshold=None,
            r0=10.0,
            c=1000.0,
            delta=0.01,
            iter_max=10,
            B=100,
            subsample_frac=0.5,
            pi_thr=0.8,
            n_lambdas=50,
            lam_ratio=1e-3,
            force_rerun_selector=args.force_rerun_selector,
        )

        print(f"[DEBUG] Xk_train shape = {Xk_train.shape}")
        print(f"[DEBUG] selected_local_indices shape = {selected_local_indices.shape}")

        assert Xk_train.shape[1] == K, (
            f"Feature dimension mismatch: got {Xk_train.shape[1]} but expected {K}"
        )

        assert len(selected_local_indices) == K, (
            f"Local index dimension mismatch: got {len(selected_local_indices)} but expected {K}"
        )

        selector_chain_paths.append(selector_path)

        # Map the local selected indices back to the original feature space.
        current_original_indices = current_original_indices[selected_local_indices]

        selected_original_indices_path = cache_dir / f"selected_original_indices_{tag}_K{K}.npy"
        np.save(selected_original_indices_path, current_original_indices.astype(int))

        selected_local_indices_path = cache_dir / f"selected_local_indices_{tag}_K{K}.npy"

        print(f"[INDEX SAVED] local indices -> {selected_local_indices_path}")
        print(f"[INDEX SAVED] original indices -> {selected_original_indices_path}")
        print(f"[DEBUG] selected_original_indices shape = {current_original_indices.shape}")
        print(
            f"[DEBUG] selected_original_indices min/max = "
            f"{current_original_indices.min()} / {current_original_indices.max()}"
        )
        print(
            f"[DEBUG] selected_original_indices unique = "
            f"{len(np.unique(current_original_indices))}"
        )

        assert len(current_original_indices) == K, (
            f"Original index dimension mismatch: got {len(current_original_indices)} but expected {K}"
        )

        assert len(np.unique(current_original_indices)) == K, (
            "Duplicate values found in selected_original_indices. "
            "Please check the chain index mapping."
        )

        assert current_original_indices.min() >= 0, (
            "Negative values found in selected_original_indices. "
            "Please check the chain index mapping."
        )

        assert current_original_indices.max() < X_train_raw.shape[1], (
            "selected_original_indices exceeds the original feature dimension. "
            f"max={current_original_indices.max()}, original_dim={X_train_raw.shape[1]}"
        )

        # Verify whether direct indexing from the original feature space can
        # reconstruct the feature matrix used by the current model.
        X_direct_from_original_indices = X_train_raw[:, current_original_indices].astype(np.float32)

        max_abs_diff_direct_index = float(
            np.max(np.abs(X_direct_from_original_indices - Xk_train))
        )

        direct_index_reconstruction_passed = bool(
            np.allclose(
                X_direct_from_original_indices,
                Xk_train,
                atol=1e-6,
                rtol=1e-5,
            )
        )

        index_verification_path = cache_dir / f"index_verification_{tag}_K{K}.json"

        index_verification = {
            "K": int(K),
            "original_feature_dim": int(X_train_raw.shape[1]),
            "selected_feature_dim": int(Xk_train.shape[1]),
            "selected_original_indices_path": str(selected_original_indices_path),
            "X_train_reconstructed_by_original_indices": "X_train_raw[:, selected_original_indices]",
            "X_train_used_by_model": str(cache_dir / f"X_train_{tag}_K{K}.npy"),
            "direct_index_reconstruction_passed": direct_index_reconstruction_passed,
            "max_abs_diff": max_abs_diff_direct_index,
            "atol": 1e-6,
            "rtol": 1e-5,
            "predictor_recommendation": (
                "Use selected_original_indices directly."
                if direct_index_reconstruction_passed
                else "Use selector_chain_paths to transform the original feature vector."
            ),
        }

        with open(index_verification_path, "w", encoding="utf-8") as f:
            json.dump(index_verification, f, ensure_ascii=False, indent=2)

        print(f"[INDEX VERIFY] passed = {direct_index_reconstruction_passed}")
        print(f"[INDEX VERIFY] max_abs_diff = {max_abs_diff_direct_index:.8g}")
        print(f"[INDEX VERIFY SAVED] -> {index_verification_path}")

        model_tag = f"tabpfn_chain_K{K}_{sampling_method_name}"

        cv_df, test_df = tabpfn_cv_and_test_on_fixed_features(
            X=Xk_train,
            y=yk_train,
            test_sets=test_k,
            out_models_dir=str(cv_models_dir),
            model_tag=model_tag,
            n_splits=args.n_splits,
            random_state=args.seed,
            device=device,
        )

        all_df = pd.concat([cv_df, test_df], ignore_index=True)

        results_csv_path = results_dir / f"results_tabpfn_chain_K{K}_{sampling_method_name}.csv"
        save_df(all_df, str(results_csv_path))

        final_model_path = train_and_save_final_tabpfn(
            X=Xk_train,
            y=yk_train,
            out_dir=str(final_model_dir),
            model_tag=model_tag,
            device=device,
            random_state=args.seed,
        )

        metadata = {
            "feature_name": feature_name,
            "sampling_method_name": sampling_method_name,
            "K": int(K),
            "original_feature_dim": int(X_train_raw.shape[1]),
            "current_feature_dim": int(Xk_train.shape[1]),
            "device_used_for_training": device,
            "random_state": args.seed,
            "threshold_for_prediction": 0.5,
            "label_map": str(label_map_path),
            "selector_path": selector_path,
            "selector_chain_paths": selector_chain_paths.copy(),
            "cv_models_dir": str(cv_models_dir),
            "final_model_path": final_model_path,
            "cache_dir": str(cache_dir),
            "results_csv": str(results_csv_path),
            "X_train_used": str(cache_dir / f"X_train_{tag}_K{K}.npy"),
            "y_train_used": str(cache_dir / f"y_train_{tag}.npy"),
            "X_test1_used": str(cache_dir / f"X_test1_{tag}_K{K}.npy"),
            "y_test1_used": str(cache_dir / f"y_test1_{tag}.npy"),
            "X_test2_used": str(cache_dir / f"X_test2_{tag}_K{K}.npy"),
            "y_test2_used": str(cache_dir / f"y_test2_{tag}.npy"),
            "selected_local_indices": str(selected_local_indices_path),
            "selected_original_indices": str(selected_original_indices_path),
            "index_verification": str(index_verification_path),
            "direct_index_reconstruction_passed": direct_index_reconstruction_passed,
            "max_abs_diff_direct_index": max_abs_diff_direct_index,
            "predictor_feature_transform_mode": (
                "direct_original_indices"
                if direct_index_reconstruction_passed
                else "selector_chain_transform_required"
            ),
            "feature_concat_order": feature_name,
            "feature_source_note": (
                "The Predictor must extract and concatenate features in the same order as training. "
                "selected_original_indices is relative to the original saprot_prott5 feature space."
            ),
            "model_save_method": "tabpfn.model_loading.save_fitted_tabpfn_model",
            "model_file_type": ".tabpfn_fit",
            "index_note": (
                "selected_local_indices is relative to the input feature space of the current chain step. "
                "selected_original_indices is relative to the original saprot_prott5 feature space. "
                "If direct_index_reconstruction_passed=True, selected_original_indices can be used directly. "
                "If False, use selector_chain_paths to reproduce the same standardization and feature selection process."
            ),
            "note": (
                "This folder contains the fixed selector, cached selected features, final TabPFN model, "
                "feature indices, label map, selector chain, and index verification results needed for "
                "fixed prediction at the current K."
            ),
        }

        save_metadata(str(k_dir / "metadata.json"), metadata)

        print(f"[DONE] K={K} finished. Folder = {k_dir}")

        # Update the chain input for the next target K.
        X_curr, y_curr, test_curr = Xk_train, yk_train, test_k

    print("\nAll chain training steps have been completed.")


if __name__ == "__main__":
    main()
