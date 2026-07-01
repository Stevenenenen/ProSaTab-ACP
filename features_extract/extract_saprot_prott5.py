# extract_saprot_prott5.py
# -*- coding: utf-8 -*-

import time
import numpy as np

from .extract_saprot import SaProtExtractor
from .extract_prott5 import ProtT5Extractor


class SaProtProtT5Extractor:
    def __init__(
        self,
        saprot_model_name: str = "westlake-repl/SaProt_1.3B_AF2",
        prott5_model_path: str = "./local/prott5",
        device=None,
        expected_saprot_dim: int = 1280,
        expected_prott5_dim: int = 1024,
        expected_total_dim: int = 2304,
    ):
        self.expected_saprot_dim = int(expected_saprot_dim)
        self.expected_prott5_dim = int(expected_prott5_dim)
        self.expected_total_dim = int(expected_total_dim)

        print("[FeatureExtractor] Loading SaProt model...")
        t0 = time.time()

        self.saprot = SaProtExtractor(
            model_name=saprot_model_name,
            max_length=257,
            use_8bit=False,
            device=device,
        )

        print(f"[FeatureExtractor] SaProt loaded. Time: {time.time() - t0:.2f}s")

        print("[FeatureExtractor] Loading ProtT5 model...")
        t1 = time.time()

        self.prott5 = ProtT5Extractor(
            model_path=prott5_model_path,
            max_length=255,
            device=device,
        )

        print(f"[FeatureExtractor] ProtT5 loaded. Time: {time.time() - t1:.2f}s")

    def extract_one(self, sequence: str) -> np.ndarray:
        print("[Feature] Extracting SaProt feature...")
        t0 = time.time()

        saprot_feat = self.saprot.extract_one(sequence)

        print(f"[Feature] SaProt feature done. Shape: {saprot_feat.shape}. Time: {time.time() - t0:.2f}s")

        print("[Feature] Extracting ProtT5 feature...")
        t1 = time.time()

        prott5_feat = self.prott5.extract_one(sequence)

        print(f"[Feature] ProtT5 feature done. Shape: {prott5_feat.shape}. Time: {time.time() - t1:.2f}s")

        self._check_component_shape(
            saprot_feat=saprot_feat,
            prott5_feat=prott5_feat,
        )

        print("[Feature] Concatenating SaProt + ProtT5 features...")
        t2 = time.time()

        feature = np.concatenate(
            [saprot_feat, prott5_feat],
            axis=1,
        ).astype(np.float32)

        self._check_total_shape(feature)

        print(f"[Feature] Concatenation done. Shape: {feature.shape}. Time: {time.time() - t2:.4f}s")

        return feature

    def extract_batch(
        self,
        sequences,
        batch_size_saprot: int = 1,
        batch_size_prott5: int = 8,
    ) -> np.ndarray:
        print(f"[BatchFeature] Extracting SaProt features for {len(sequences)} sequences...")
        t0 = time.time()

        saprot_feat = self.saprot.extract_batch(
            sequences,
            batch_size=batch_size_saprot,
        )

        print(f"[BatchFeature] SaProt batch done. Shape: {saprot_feat.shape}. Time: {time.time() - t0:.2f}s")

        print(f"[BatchFeature] Extracting ProtT5 features for {len(sequences)} sequences...")
        t1 = time.time()

        prott5_feat = self.prott5.extract_batch(
            sequences,
            batch_size=batch_size_prott5,
        )

        print(f"[BatchFeature] ProtT5 batch done. Shape: {prott5_feat.shape}. Time: {time.time() - t1:.2f}s")

        if saprot_feat.shape[0] != prott5_feat.shape[0]:
            raise ValueError(
                "SaProt and ProtT5 sample sizes are inconsistent: "
                f"SaProt={saprot_feat.shape}, ProtT5={prott5_feat.shape}"
            )

        self._check_component_shape(
            saprot_feat=saprot_feat,
            prott5_feat=prott5_feat,
        )

        print("[BatchFeature] Concatenating batch features...")
        t2 = time.time()

        feature = np.concatenate(
            [saprot_feat, prott5_feat],
            axis=1,
        ).astype(np.float32)

        self._check_total_shape(feature)

        print(f"[BatchFeature] Concatenation done. Shape: {feature.shape}. Time: {time.time() - t2:.4f}s")

        return feature

    def _check_component_shape(
        self,
        saprot_feat: np.ndarray,
        prott5_feat: np.ndarray,
    ):
        if saprot_feat.ndim != 2:
            raise ValueError(
                f"SaProt feature must be a 2D array, got shape={saprot_feat.shape}"
            )

        if prott5_feat.ndim != 2:
            raise ValueError(
                f"ProtT5 feature must be a 2D array, got shape={prott5_feat.shape}"
            )

        if saprot_feat.shape[1] != self.expected_saprot_dim:
            raise ValueError(
                f"Unexpected SaProt feature dimension: "
                f"got {saprot_feat.shape[1]}, expected {self.expected_saprot_dim}"
            )

        if prott5_feat.shape[1] != self.expected_prott5_dim:
            raise ValueError(
                f"Unexpected ProtT5 feature dimension: "
                f"got {prott5_feat.shape[1]}, expected {self.expected_prott5_dim}"
            )

    def _check_total_shape(self, feature: np.ndarray):
        if feature.ndim != 2:
            raise ValueError(
                f"Concatenated feature must be a 2D array, got shape={feature.shape}"
            )

        if feature.shape[1] != self.expected_total_dim:
            raise ValueError(
                f"Unexpected concatenated feature dimension: "
                f"got {feature.shape[1]}, expected {self.expected_total_dim}."
            )


def extract_saprot_prott5_feature(
    sequence: str,
    saprot_model_name: str = "westlake-repl/SaProt_1.3B_AF2",
    prott5_model_path: str = "./local/prott5",
    device=None,
) -> np.ndarray:
    extractor = SaProtProtT5Extractor(
        saprot_model_name=saprot_model_name,
        prott5_model_path=prott5_model_path,
        device=device,
    )

    return extractor.extract_one(sequence)