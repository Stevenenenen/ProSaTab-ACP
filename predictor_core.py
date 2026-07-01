# predictor_core.py
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path

import numpy as np
import torch

from stable_lasso_selector import StableLassoSelector
from features_extract.extract_saprot_prott5 import SaProtProtT5Extractor

try:
    from tabpfn.model_loading import load_fitted_tabpfn_model
except ImportError:
    load_fitted_tabpfn_model = None


class ACPPredictor:
    def __init__(
        self,
        model_dir="models",
        feature_device="cpu",
        tabpfn_device="cuda",
    ):
        self.root_dir = Path(__file__).resolve().parent
        self.model_dir = self.root_dir / model_dir

        self.feature_device = self._check_device(feature_device)
        self.tabpfn_device = self._check_device(tabpfn_device)

        print(f"[DEVICE] Feature extractor device: {self.feature_device}")
        print(f"[DEVICE] TabPFN device: {self.tabpfn_device}")

        config_path = self.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        label_map_path = self.model_dir / self.config["label_map"]
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        self.original_dim = int(self.config["original_feature_dim"])
        self.selected_dim = int(self.config["selected_feature_dim"])
        self.final_k = int(self.config["final_k"])
        self.threshold = float(self.config.get("threshold_for_prediction", 0.5))

        self.load_feature_extractor()
        self.load_selector_chain()
        self.load_tabpfn_model()

        print("[LOAD] Predictor ready.")

    @staticmethod
    def _check_device(device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA is not available. Falling back to CPU.")
            return "cpu"

        return device

    def load_feature_extractor(self):
        prott5_model_path = Path(self.config["prott5"]["model_path"])

        if not prott5_model_path.is_absolute():
            prott5_model_path = self.root_dir / prott5_model_path

        if not prott5_model_path.exists():
            raise FileNotFoundError(f"ProtT5 model path not found: {prott5_model_path}")

        print("[LOAD] Initializing SaProt + ProtT5 feature extractor...")
        t0 = time.time()

        self.feature_extractor = SaProtProtT5Extractor(
            saprot_model_name=self.config["saprot"]["model_name"],
            prott5_model_path=str(prott5_model_path),
            device=self.feature_device,
        )

        print(f"[LOAD] Feature extractor loaded. Time: {time.time() - t0:.2f}s")

    def load_selector_chain(self):
        print("[LOAD] Loading Stable LASSO selector chain...")
        t0 = time.time()

        self.selectors = []

        for i, rel_path in enumerate(self.config["preprocessor_chain"], start=1):
            selector_path = self.model_dir / rel_path

            if not selector_path.exists():
                raise FileNotFoundError(f"Selector file not found: {selector_path}")

            selector = StableLassoSelector.load(str(selector_path))
            self.selectors.append(selector)

            print(f"[LOAD] Selector step {i:02d} loaded: {selector_path.name}")

        print(
            f"[LOAD] Selector chain loaded. "
            f"Total steps: {len(self.selectors)}. "
            f"Time: {time.time() - t0:.2f}s"
        )

    def load_tabpfn_model(self):
        if load_fitted_tabpfn_model is None:
            raise ImportError(
                "load_fitted_tabpfn_model is not available in the current TabPFN version."
            )

        final_model_path = self.model_dir / self.config["final_model_path"]

        if not final_model_path.exists():
            raise FileNotFoundError(f"Final model not found: {final_model_path}")

        print(f"[LOAD] Loading final TabPFN model on {self.tabpfn_device}...")
        t0 = time.time()

        if self.tabpfn_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            self.model = load_fitted_tabpfn_model(
                str(final_model_path),
                device=self.tabpfn_device,
            )
        except TypeError:
            print("[WARNING] Current TabPFN version does not support device argument.")
            self.model = load_fitted_tabpfn_model(str(final_model_path))

        if self.tabpfn_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"[LOAD] Final TabPFN model loaded. Time: {time.time() - t0:.2f}s")

        if hasattr(self.model, "devices_"):
            print(f"[DEBUG] TabPFN devices_: {self.model.devices_}")

        if hasattr(self.model, "n_features_in_"):
            print(f"[DEBUG] TabPFN n_features_in_: {self.model.n_features_in_}")

        if hasattr(self.model, "class_counts_"):
            print(f"[DEBUG] TabPFN class_counts_: {self.model.class_counts_}")

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        sequence = sequence.strip().upper()
        sequence = "".join(sequence.split())

        if not sequence:
            raise ValueError("Empty sequence.")

        return sequence

    def extract_feature(self, sequence: str) -> np.ndarray:
        x_full = self.feature_extractor.extract_one(sequence)

        if x_full.ndim != 2:
            raise ValueError(f"Invalid feature shape: {x_full.shape}")

        if x_full.shape[1] != self.original_dim:
            raise ValueError(
                f"Original feature dimension mismatch: "
                f"got {x_full.shape[1]}, expected {self.original_dim}."
            )

        return x_full.astype(np.float32)

    def transform_feature(self, x_full: np.ndarray) -> np.ndarray:
        x = x_full.astype(np.float32)

        for i, selector in enumerate(self.selectors, start=1):
            t0 = time.time()
            x = selector.transform(x)
            print(
                f"[SELECTOR] Step {i:02d}/{len(self.selectors)} done. "
                f"Shape: {x.shape}. Time: {time.time() - t0:.4f}s"
            )

        if x.shape[1] != self.selected_dim:
            raise ValueError(
                f"Selected feature dimension mismatch: "
                f"got {x.shape[1]}, expected {self.selected_dim}."
            )

        return x.astype(np.float32)

    def predict_tabpfn(self, x_selected: np.ndarray) -> float:
        x_selected = np.asarray(x_selected, dtype=np.float32)

        print("[DEBUG] Input to TabPFN:", x_selected.shape, x_selected.dtype)
        print(f"[DEBUG] TabPFN prediction device: {self.tabpfn_device}")

        if self.tabpfn_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        t0 = time.time()

        prob = self.model.predict_proba(x_selected)

        if self.tabpfn_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - t0

        print(f"[DEBUG] Raw predict_proba output: {prob}")
        print(f"[DEBUG] TabPFN predict_proba elapsed: {elapsed:.2f}s")

        return float(prob[0, 1])

    def predict(self, sequence: str) -> dict:
        total_t0 = time.time()

        sequence = self.clean_sequence(sequence)

        print("\n" + "=" * 70)
        print("[PREDICT] New sequence received.")
        print(f"[PREDICT] Sequence length: {len(sequence)}")
        print("=" * 70)

        print("[1/4] Extracting SaProt + ProtT5 features...")
        t0 = time.time()
        x_full = self.extract_feature(sequence)
        print(f"[1/4] Full feature extracted: {x_full.shape}. Time: {time.time() - t0:.2f}s")

        print("[2/4] Applying Stable LASSO selector chain...")
        t1 = time.time()
        x_selected = self.transform_feature(x_full)
        print(f"[2/4] Selected feature ready: {x_selected.shape}. Time: {time.time() - t1:.2f}s")

        print("[3/4] Running TabPFN prediction...")
        t2 = time.time()
        prob_acp = self.predict_tabpfn(x_selected)
        print(f"[3/4] TabPFN prediction done. Time: {time.time() - t2:.2f}s")

        pred_id = int(prob_acp >= self.threshold)
        pred_label = self.label_map[str(pred_id)]

        print("[4/4] Prediction completed.")
        print(f"[RESULT] Prediction: {pred_label}")
        print(f"[RESULT] Probability ACP: {prob_acp:.4f}")
        print(f"[RESULT] Total prediction time: {time.time() - total_t0:.2f}s")
        print("=" * 70 + "\n")

        return {
            "sequence": sequence,
            "prediction": pred_label,
            "probability_acp": prob_acp,
            "threshold": self.threshold,
            "original_dim": int(x_full.shape[1]),
            "selected_dim": int(x_selected.shape[1]),
            "final_k": self.final_k,
        }