"""
RoboSense: Explainable Failure Prediction for Robotic Arms
Trains and compares an XGBoost (feature-based) model against an LSTM
(raw-sequence) model, handles class imbalance with SMOTE, and produces
SHAP explainability for the winning model.

Usage:
    python train.py --lp data/lp1.data.txt
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

from data_loader import load_dataset
from features import extract_features


def train_xgboost(X_feat, y_enc, class_names):
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    # Handle class imbalance -- only oversample the training split
    min_class_count = np.min(np.bincount(y_train))
    k_neighbors = max(1, min(5, min_class_count - 1))
    if min_class_count > 1:
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
    }


def train_lstm(X_raw, y_enc, n_classes, class_names):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    # normalize per-channel using train stats
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    model = models.Sequential([
        layers.Input(shape=(X_raw.shape[1], X_raw.shape[2])),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_n, y_train, epochs=60, batch_size=8, verbose=0, validation_split=0.2)

    y_pred_probs = model.predict(X_test_n, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "model": model,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
    }


def run_shap(xgb_result, feature_names, class_names):
    import shap

    model = xgb_result["model"]
    X_test = xgb_result["X_test"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # mean |SHAP| per feature, averaged across classes if multi-class
    if isinstance(shap_values, list):
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importance = np.abs(shap_values).mean(axis=(0, -1)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    top_idx = np.argsort(importance)[::-1][:10]
    top_features = [(feature_names[i], float(importance[i])) for i in top_idx]
    return top_features


def main(lp_path):
    print(f"\n=== Loading {lp_path} ===")
    X_raw, y_raw = load_dataset(lp_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    print("\n=== Extracting features (for XGBoost) ===")
    feat_df = extract_features(X_raw)
    feature_names = feat_df.columns.tolist()
    X_feat = feat_df.values

    print("\n=== Training XGBoost (feature-based) ===")
    xgb_result = train_xgboost(X_feat, y_enc, class_names)
    print(f"XGBoost macro-F1: {xgb_result['macro_f1']:.3f}")
    print(pd.DataFrame(xgb_result["report"]).T.round(3))

    print("\n=== Training LSTM (raw sequence) ===")
    lstm_result = train_lstm(X_raw, y_enc, len(class_names), class_names)
    print(f"LSTM macro-F1: {lstm_result['macro_f1']:.3f}")
    print(pd.DataFrame(lstm_result["report"]).T.round(3))

    print("\n=== Model comparison ===")
    winner = "XGBoost" if xgb_result["macro_f1"] >= lstm_result["macro_f1"] else "LSTM"
    print(f"XGBoost macro-F1: {xgb_result['macro_f1']:.3f}")
    print(f"LSTM macro-F1:    {lstm_result['macro_f1']:.3f}")
    print(f"Winner: {winner}")

    print("\n=== SHAP explainability (on XGBoost) ===")
    top_features = run_shap(xgb_result, feature_names, class_names)
    for name, val in top_features:
        print(f"  {name:20s}: {val:.4f}")

    # Save a summary report for the slides/report
    summary = {
        "dataset": lp_path,
        "n_instances": int(X_raw.shape[0]),
        "classes": class_names,
        "xgboost_macro_f1": xgb_result["macro_f1"],
        "xgboost_confusion_matrix": xgb_result["confusion_matrix"],
        "lstm_macro_f1": lstm_result["macro_f1"],
        "lstm_confusion_matrix": lstm_result["confusion_matrix"],
        "winner": winner,
        "top_shap_features": top_features,
    }
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", default="data/lp1.data.txt")
    args = parser.parse_args()
    main(args.lp)
