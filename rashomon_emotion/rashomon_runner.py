"""
Rashomon Runner — Unified training, evaluation, and Rashomon loop.
"""

import json
import time
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from .model import EEGGNN
from .utils import save_pickle, load_pickle
from .preprocessing import load_data, normalize_trials
from .features import extract_wavelet_features, extract_dwt_subbands
from .graph_utils import build_connectivity_graph


# ---------------------- TRAINING ---------------------- #
def _set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN for reproducibility (may reduce perf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, features, labels, epochs=50, lr=1e-3, device="cpu", seed=None, history_path=None, log_interval=10):
    """
    Train EEG model. Supports optional deterministic seeding and history logging.

    Args:
        model: torch.nn.Module
        features: array-like, shape (n_samples, n_features)
        labels: array-like, shape (n_samples,)
        epochs: int
        lr: float
        device: 'cpu' or 'cuda'
        seed: optional int for reproducibility
        history_path: optional path to save training history JSON
        log_interval: epochs between printing logs

    Returns:
        model (trained)
    """
    _set_seed(seed)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(labels), dtype=torch.long).to(device)

    history = {"loss": [], "accuracy": [], "epoch_time_s": []}
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # record metrics (full-batch)
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y).float().mean().item()

        history["loss"].append(float(loss.item()))
        history["accuracy"].append(float(acc))
        epoch_time = time.time() - epoch_start
        history["epoch_time_s"].append(epoch_time)

        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f} Acc: {acc:.4f} (t={epoch_time:.2f}s)")

    total_time = time.time() - start_time
    history_meta = {
        "epochs": epochs,
        "lr": lr,
        "device": str(device),
        "seed": seed,
        "total_time_s": total_time
    }
    history.update(history_meta)

    if history_path:
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
        except Exception:
            pass
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

    return model


# ---------------------- EVALUATION ---------------------- #
def evaluate_model(model_path, data_path, device="cpu"):
    """
    Evaluate model from saved checkpoint.
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint.get("input_dim", None)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    output_dim = checkpoint.get("output_dim", 2)

    model = EEGGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Load and preprocess
    eeg, labels = load_data(data_path)
    eeg = normalize_trials(eeg)
    # default to wavelet features; caller can precompute different features if needed
    features = [extract_wavelet_features(trial) for trial in eeg]

    X = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    y_true = np.array(labels)

    with torch.no_grad():
        outputs = model(X)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return {"accuracy": acc, "f1_score": f1}


# ---------------------- RASHOMON LOOP ---------------------- #
def run_rashomon(data_path, param_grid, epochs=50, lr=1e-3, device="cpu"):
    """
    Run Rashomon loop — train multiple models with different hyperparameters.
    """
    eeg, labels = load_deap_data(data_path)
    eeg = normalize_trials(eeg)
    features = [extract_wavelet_features(trial) for trial in eeg]

    results = []
    for params in param_grid:
        print(f"Training with params: {params}")
        model = EEGGNN(
            input_dim=len(features[0]),
            hidden_dim=params.get("hidden_dim", 64),
            output_dim=params.get("output_dim", 2)
        )

        trained_model = train_model(model, features, labels, epochs=epochs, lr=lr, device=device)

        # Save checkpoint
        checkpoint = {
            "state_dict": trained_model.state_dict(),
            "input_dim": len(features[0]),
            "hidden_dim": params.get("hidden_dim", 64),
            "output_dim": params.get("output_dim", 2)
        }
        model_path = f"rashomon_model_{params.get('hidden_dim',64)}.pt"
        torch.save(checkpoint, model_path)

        # Evaluate
        metrics = evaluate_model(model_path, data_path, device=device)
        print(f"Metrics: {metrics}")

        results.append({"params": params, "metrics": metrics})

    save_pickle(results, "rashomon_results.pkl")
    return results
