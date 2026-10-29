"""
Comparative ML and explainability study for the Wisconsin Breast Cancer dataset.

Models:
  - Logistic Regression (scikit-learn)
  - Random Forest (scikit-learn)
  - XGBoost
  - MLP (PyTorch)
  - 1D-CNN (PyTorch)
  - Transformer encoder (PyTorch)

Outputs are written to OUTPUT_DIR:
  - model_metrics.csv / model_metrics_table.png
  - per-model confusion matrices and ROC curves
  - SHAP summary plots and LIME instance explanations
  - run_metadata.json with dataset/split/configuration details

Research notes:
  * Positive class is explicitly malignant (label 1).
  * The test set is held out and is not used for early stopping or explanations'
    background selection. SHAP/LIME explanations use training data as background.
  * Cross-validation is used on the training split for a less split-sensitive
    comparison of the classical models. The final metrics below are still from
    one untouched test split; neural-network CV is not run by this script.
  * CNN/Transformer treat the ordered feature vector as a sequence. Since these
    are tabular measurements, this inductive bias is experimental rather than
    inherently appropriate; compare results rather than assuming they should win.
  * This is educational/research code, not a clinical diagnostic system.
"""

import os
import json
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, make_scorer
)

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------------- Configuration ----------------
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
NN_EPOCHS = 100
NN_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
SHAP_BACKGROUND_SIZE = 40
SHAP_EXPLAIN_SIZE = 20
SHAP_NSAMPLES = 100
LIME_INSTANCE_INDEX = 0
LIME_NUM_FEATURES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Data ----------------
def load_data():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    # sklearn's built-in encoding is 0=malignant, 1=benign.
    # Re-map so the positive class (1) is malignant for clinically relevant metrics.
    y = (data.target == 0).astype(np.int64)
    feature_names = list(data.feature_names)
    class_names = ["Benign", "Malignant"]  # index 0, index 1
    return X, y, feature_names, class_names


def split_and_scale(X, y):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    return X_train, X_test, y_train, y_test, scaler


# ---------------- Metrics / prediction helpers ----------------
def specificity_score(y_true, y_pred):
    # With positive class=malignant (1), specificity is correctly measured for benign (0).
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def evaluate_model(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Malignant": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_Sensitivity_Malignant": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Specificity_Benign": specificity_score(y_true, y_pred),
        "F1_Malignant": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ROC_AUC_Malignant": roc_auc_score(y_true, y_proba),
    }


def sklearn_predict_proba(model, data):
    return model.predict_proba(np.asarray(data))  # columns [benign, malignant]


def torch_predict_proba(model, data, reshape=False, batch_size=256):
    """Return two-column probabilities [benign, malignant] for SHAP/LIME."""
    data = np.asarray(data, dtype=np.float32)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[start:start + batch_size]).to(DEVICE)
            if reshape:
                batch = batch.unsqueeze(1)  # (N, channels=1, sequence_length=30)
            logits = model(batch).reshape(-1)
            p_malignant = torch.sigmoid(logits)
            p_benign = 1.0 - p_malignant
            outputs.append(torch.stack([p_benign, p_malignant], dim=1).cpu().numpy())
    return np.vstack(outputs) if outputs else np.empty((0, 2), dtype=np.float32)


def get_sklearn_predictions(model, X):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba


def get_torch_predictions(model, X, reshape=False):
    proba = torch_predict_proba(model, X, reshape=reshape)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba


# ---------------- Plotting ----------------
def save_metrics_table(results, path):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, "model_metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.25), len(df) * 0.6 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "model_metrics_table.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_confusion_matrix(y_true, y_pred, name, path):
    # Labels are consistently [0=Benign, 1=Malignant].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"confusion_matrix_{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, name, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=1)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (malignant)")
    ax.set_title(f"ROC Curve - {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"roc_curve_{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- Classical ML models ----------------
def make_logistic_regression():
    return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced"
    )


def make_xgboost():
    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )


# ---------------- PyTorch models ----------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)  # logits; sigmoid is applied only for probabilities
        )

    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Expected shape: (batch, channels=1, sequence_length=input_dim)
        return self.classifier(self.features(x))


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim, d_model=32, num_heads=4, ff_dim=64, dropout=0.10):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.feature_projection = nn.Linear(1, d_model)
        # A learnable feature-position embedding distinguishes feature columns.
        self.position_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, 1, input_dim); convert to (batch, features, 1)
        x = x.transpose(1, 2)
        x = self.feature_projection(x) + self.position_embedding
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


def train_torch_model(model, X_train, y_train, reshape=False,
                      max_epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE,
                      patience=EARLY_STOPPING_PATIENCE):
    """
    Creates a validation split from training data only, uses early stopping,
    and restores the best validation-loss weights. Test data is never passed here.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    def to_tensor(x):
        tensor = torch.as_tensor(x, dtype=torch.float32)
        if reshape:
            tensor = tensor.unsqueeze(1)  # Conv1d layout: (N, 1, features)
        return tensor

    train_ds = TensorDataset(
        to_tensor(X_fit),
        torch.as_tensor(y_fit, dtype=torch.float32).reshape(-1, 1)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = to_tensor(X_val).to(DEVICE)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------- SHAP ----------------
def normalize_shap_values(shap_values):
    """Return the SHAP values for malignant probability as (samples, features)."""
    # Recent SHAP versions may return an Explanation object.
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        # Binary classification commonly returns one array per class.
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Possible shape (samples, features, classes) or (samples, classes, features).
        if values.shape[1] == 30 and values.shape[2] == 2:
            values = values[:, :, 1]
        elif values.shape[1] == 2 and values.shape[2] == 30:
            values = values[:, 1, :]
        else:
            values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def save_shap_summary(values, X_explain, feature_names, name, path):
    shap.summary_plot(
        values, X_explain, feature_names=feature_names, show=False,
        max_display=min(20, len(feature_names)),
        rng = np.random.default_rng(RANDOM_STATE)
    )
    plt.title(f"SHAP Summary - {name} (malignant probability)")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"shap_summary_{name}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def explain_shap_tree(model, X_background, X_explain, feature_names, name, path):
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_explain)
    values = normalize_shap_values(raw)
    save_shap_summary(values, X_explain, feature_names, name, path)


def explain_shap_linear(model, X_background, X_explain, feature_names, name, path):
    explainer = shap.LinearExplainer(model, X_background)
    raw = explainer.shap_values(X_explain)
    values = normalize_shap_values(raw)
    save_shap_summary(values, X_explain, feature_names, name, path)


def explain_shap_kernel(predict_fn, X_background, X_explain, feature_names, name, path):
    background_n = min(SHAP_BACKGROUND_SIZE, len(X_background))
    explain_n = min(SHAP_EXPLAIN_SIZE, len(X_explain))
    background = shap.sample(X_background, background_n, random_state=RANDOM_STATE)
    sample_test = X_explain[:explain_n]
    explainer = shap.KernelExplainer(lambda data: predict_fn(data)[:, 1], background)
    raw = explainer.shap_values(sample_test, nsamples=SHAP_NSAMPLES)
    values = normalize_shap_values(raw)
    save_shap_summary(values, sample_test, feature_names, name, path)


# ---------------- LIME ----------------
def explain_lime(predict_fn, X_train, X_test, feature_names, class_names, name, path,
                 instance_idx=LIME_INSTANCE_INDEX):
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )
    idx = min(max(instance_idx, 0), len(X_test) - 1)
    exp = explainer.explain_instance(
        X_test[idx], predict_fn, num_features=min(LIME_NUM_FEATURES, len(feature_names))
    )
    fig = exp.as_pyplot_figure()
    fig.suptitle(f"LIME Explanation - {name} (instance {idx}; 1=malignant)")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"lime_explanation_{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- Cross-validation ----------------
def run_classical_cv(X_train_raw, y_train):
    """
    Stratified CV on training data only. Scaling is inside the LR pipeline to avoid
    fold leakage. Tree models do not need scaling, but are evaluated on raw features.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_malignant": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_malignant": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1_malignant": make_scorer(f1_score, pos_label=1, zero_division=0),
        "roc_auc_malignant": "roc_auc"
    }

    models_to_cv = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", make_logistic_regression())
        ]),
        "RandomForest": make_random_forest(),
        "XGBoost": make_xgboost()
    }

    rows = []
    for name, estimator in models_to_cv.items():
        scores = cross_validate(
            estimator, X_train_raw, y_train, cv=cv, scoring=scoring,
            n_jobs=1, return_train_score=False
        )
        row = {"Model": name}
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            row[f"CV_{metric}_mean"] = float(np.mean(vals))
            row[f"CV_{metric}_std"] = float(np.std(vals, ddof=1))
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "classical_cv_metrics.csv"), index=False)


def main():
    print(f"Using device: {DEVICE}")
    X, y, feature_names, class_names = load_data()

    # Split raw data first; fit scaler only on training observations.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Training-only CV for classical models, with fold-local preprocessing.
    run_classical_cv(X_train_raw, y_train)

    results = []

    # Classical models: LR uses scaled features; tree-based models use raw features.
    classical = [
        ("LogisticRegression", make_logistic_regression(), X_train, X_test, True),
        ("RandomForest", make_random_forest(), X_train_raw, X_test_raw, False),
        ("XGBoost", make_xgboost(), X_train_raw, X_test_raw, False)
    ]

    for name, model, Xtr, Xte, _scaled in classical:
        model.fit(Xtr, y_train)
        y_pred, y_proba = get_sklearn_predictions(model, Xte)
        results.append(evaluate_model(name, y_test, y_pred, y_proba))
        plot_confusion_matrix(y_test, y_pred, name, OUTPUT_DIR)
        plot_roc_curve(y_test, y_proba, name, OUTPUT_DIR)

        # SHAP and LIME operate on the same feature representation used by this model.
        if name == "LogisticRegression":
            explain_shap_linear(model, X_train, X_test, feature_names, name, OUTPUT_DIR)
            lime_train, lime_test = X_train, X_test
        else:
            explain_shap_tree(model, X_train_raw, X_test_raw, feature_names, name, OUTPUT_DIR)
            lime_train, lime_test = X_train_raw, X_test_raw

        explain_lime(
            model.predict_proba, lime_train, lime_test, feature_names, class_names,
            name, OUTPUT_DIR
        )

    # Deep learning models: all use standardized features.
    torch_models = [
        ("MLP", MLP(X_train.shape[1]), False),
        ("1D-CNN", CNN1D(X_train.shape[1]), True),
        ("Transformer", TransformerEncoderClassifier(X_train.shape[1]), True)
    ]

    for name, model, reshape in torch_models:
        print(f"Training {name}...")
        model = train_torch_model(model, X_train, y_train, reshape=reshape)
        y_pred, y_proba = get_torch_predictions(model, X_test, reshape=reshape)
        results.append(evaluate_model(name, y_test, y_pred, y_proba))
        plot_confusion_matrix(y_test, y_pred, name, OUTPUT_DIR)
        plot_roc_curve(y_test, y_proba, name, OUTPUT_DIR)

        predict_fn = lambda data, m=model, r=reshape: torch_predict_proba(m, data, reshape=r)
        explain_shap_kernel(
            predict_fn, X_train, X_test, feature_names, name, OUTPUT_DIR
        )
        explain_lime(
            predict_fn, X_train, X_test, feature_names, class_names, name, OUTPUT_DIR
        )

        # Save model weights for reproducibility/reuse.
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name.lower().replace('-', '_')}_state_dict.pt"))

    # Save consolidated held-out test metrics.
    df = save_metrics_table(results, OUTPUT_DIR)
    print("\nHeld-out test metrics (positive class = malignant):")
    print(df.to_string(index=False))

    metadata = {
        "dataset": "sklearn Breast Cancer Wisconsin Diagnostic",
        "n_samples": int(len(y)),
        "n_features": int(len(feature_names)),
        "class_mapping": {"0": "Benign", "1": "Malignant"},
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "device": str(DEVICE),
        "models": [row["Model"] for row in results],
        "notes": [
            "Test set was held out from training and classical cross-validation.",
            "Classical CV metrics are saved separately in classical_cv_metrics.csv.",
            "SHAP/LIME explanations are post-hoc and do not establish clinical validity.",
            "CNN and Transformer treat feature columns as a sequence; this is an experimental inductive bias."
        ]
    }
    with open(os.path.join(OUTPUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll outputs saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    # Avoid excessive CPU thread contention on small systems.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    main()
