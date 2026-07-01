# check_tabpfn_only.py
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tabpfn.model_loading import load_fitted_tabpfn_model


def sync_cuda(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_config(model_dir: Path):
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_final_model(model_dir: Path, config: dict):
    if "final_model_path" in config:
        model_path = model_dir / config["final_model_path"]
        if model_path.exists():
            return model_path

    candidates = list(model_dir.rglob("*.tabpfn_fit"))
    if not candidates:
        raise FileNotFoundError(f"No .tabpfn_fit file found under: {model_dir}")

    if len(candidates) > 1:
        print("[WARNING] Multiple .tabpfn_fit files found. Using the first one:")
        for p in candidates:
            print("  -", p)

    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--input-npy", type=str, default=None)

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    model_dir = root_dir / args.model_dir

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Use CPU instead.")
        device = "cpu"

    print("=" * 80)
    print("[ENV]")
    print("root_dir:", root_dir)
    print("model_dir:", model_dir)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("gpu memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    config = load_config(model_dir)

    selected_dim = int(config.get("selected_feature_dim", config.get("final_k", 564)))
    final_model_path = find_final_model(model_dir, config)

    print("=" * 80)
    print("[MODEL]")
    print("selected_dim:", selected_dim)
    print("final_model_path:", final_model_path)

    if device == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    try:
        model = load_fitted_tabpfn_model(str(final_model_path), device=device)
    except TypeError:
        print("[WARNING] This TabPFN version does not support device argument.")
        model = load_fitted_tabpfn_model(str(final_model_path))

    sync_cuda(device)
    print("model loading time:", f"{time.time() - t0:.2f}s")

    if hasattr(model, "device"):
        print("model.device:", model.device)
    if hasattr(model, "devices_"):
        print("model.devices_:", model.devices_)
    if hasattr(model, "n_features_in_"):
        print("model.n_features_in_:", model.n_features_in_)
    if hasattr(model, "class_counts_"):
        print("model.class_counts_:", model.class_counts_)

    print("=" * 80)
    print("[INPUT]")

    if args.input_npy:
        input_path = Path(args.input_npy)
        if not input_path.is_absolute():
            input_path = root_dir / input_path

        x = np.load(input_path).astype(np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        print("input:", input_path)
    else:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1, selected_dim)).astype(np.float32)
        print("input: random")

    print("x shape:", x.shape)
    print("x dtype:", x.dtype)

    if x.shape[1] != selected_dim:
        print(f"[WARNING] x dimension is {x.shape[1]}, but selected_dim is {selected_dim}")

    print("=" * 80)
    print("[PREDICT_PROBA SPEED TEST]")

    times = []

    for i in range(args.repeat):
        if device == "cuda":
            torch.cuda.empty_cache()

        sync_cuda(device)
        t1 = time.time()

        prob = model.predict_proba(x)

        sync_cuda(device)
        elapsed = time.time() - t1
        times.append(elapsed)

        print(f"run {i + 1}: {elapsed:.2f}s, prob={prob}")

    print("=" * 80)
    print("[SUMMARY]")
    print("device:", device)
    print("times:", [round(t, 2) for t in times])
    print("first:", f"{times[0]:.2f}s")

    if len(times) > 1:
        print("mean after first:", f"{np.mean(times[1:]):.2f}s")

    print("[DONE]")


if __name__ == "__main__":
    main()