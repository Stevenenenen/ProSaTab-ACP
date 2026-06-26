# -*- coding: utf-8 -*-
"""MDNDO-ENN resampling module for ACP classification."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


ArrayLike = np.ndarray
SamplingStrategy = Dict[int, int]


def over_mdndo(
    data: ArrayLike,
    labels: ArrayLike,
    sampling_strategy: SamplingStrategy,
    noise_scale: float = 0.05,
    random_state: int | None = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Generate synthetic samples with Gaussian perturbation."""
    rng = np.random.default_rng(random_state)

    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels).ravel()

    if data.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("data and labels must have the same number of samples.")

    resampled_data = []
    resampled_labels = []

    for label, target_total in sampling_strategy.items():
        class_data = data[labels == label]
        original_count = class_data.shape[0]

        if original_count == 0:
            raise ValueError(f"Class {label} has no samples.")
        if target_total < original_count:
            raise ValueError(
                f"target_total for class {label} is smaller than the original count."
            )

        resampled_data.append(class_data)
        resampled_labels.extend([label] * original_count)

        new_samples_needed = target_total - original_count
        if new_samples_needed <= 0:
            continue

        generated_samples = []
        sample_idx = 0

        while len(generated_samples) < new_samples_needed:
            sample = class_data[sample_idx % original_count].astype(np.float64)
            covariance = np.diag(noise_scale * np.square(sample)).astype(np.float64)
            new_sample = rng.multivariate_normal(sample, covariance, 1)[0]
            generated_samples.append(new_sample)
            resampled_labels.append(label)
            sample_idx += 1

        resampled_data.append(np.asarray(generated_samples, dtype=np.float64))

    resampled_data = np.vstack(resampled_data).astype(np.float32)
    resampled_labels = np.asarray(resampled_labels)

    return resampled_data, resampled_labels


def edited_nearest_neighbors(
    X: ArrayLike,
    y: ArrayLike,
    sampling_strategy: SamplingStrategy | None = None,
    k: int = 3,
) -> Tuple[ArrayLike, ArrayLike]:
    """Apply ENN undersampling."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if k < 1 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")

    if sampling_strategy is None:
        sampling_strategy = {label: int(np.sum(y == label)) for label in np.unique(y)}

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    keep_indices = []
    class_counts = {label: 0 for label in np.unique(y)}

    for i in range(X.shape[0]):
        _, indices = nn.kneighbors(X[i].reshape(1, -1))
        nearest_labels = y[indices[0]]
        current_label = y[i]

        majority_supported = np.sum(nearest_labels == current_label) > k // 2
        under_target = class_counts[current_label] < sampling_strategy.get(
            current_label,
            int(np.sum(y == current_label)),
        )

        if majority_supported and under_target:
            keep_indices.append(i)
            class_counts[current_label] += 1

    return X[keep_indices], y[keep_indices]


def mdndo_enn_resample(
    X: ArrayLike,
    y: ArrayLike,
    over_sampling_strategy: SamplingStrategy,
    under_sampling_strategy: SamplingStrategy | None = None,
    k: int = 3,
    noise_scale: float = 0.05,
    random_state: int | None = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Run MDNDO oversampling followed by ENN undersampling."""
    X_over, y_over = over_mdndo(
        data=X,
        labels=y,
        sampling_strategy=over_sampling_strategy,
        noise_scale=noise_scale,
        random_state=random_state,
    )

    return edited_nearest_neighbors(
        X=X_over,
        y=y_over,
        sampling_strategy=under_sampling_strategy,
        k=k,
    )


# Compatible names for older scripts.
Over_MDNDO = over_mdndo
UnderENN = edited_nearest_neighbors

