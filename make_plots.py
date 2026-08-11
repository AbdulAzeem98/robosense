"""
RoboSense: Generates confusion matrix and SHAP summary plots for slides.
Run this AFTER train.py has produced results_summary.json.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from data_loader import load_dataset
from features import extract_features
from train import train_xgboost


def plot_confusion_matrix(cm, class_names, title, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_shap_bar(top_features, out_path):
    names = [n for n, _ in top_features]
    vals = [v for _, v in top_features]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(names[::-1], vals[::-1], color="#2a9d8f")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top features driving failure predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_model_comparison(xgb_f1, lstm_f1, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(["XGBoost\n(features)", "LSTM\n(raw sequence)"], [xgb_f1, lstm_f1],
                   color=["#2a9d8f", "#e76f51"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Model comparison")
    for bar, val in zip(bars, [xgb_f1, lstm_f1]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    with open("results_summary.json") as f:
        summary = json.load(f)

    class_names = summary["classes"]
    plot_confusion_matrix(summary["xgboost_confusion_matrix"], class_names,
                           "XGBoost Confusion Matrix", "plot_xgb_confusion.png")
    plot_confusion_matrix(summary["lstm_confusion_matrix"], class_names,
                           "LSTM Confusion Matrix", "plot_lstm_confusion.png")
    plot_shap_bar(summary["top_shap_features"], "plot_shap_importance.png")
    plot_model_comparison(summary["xgboost_macro_f1"], summary["lstm_macro_f1"],
                           "plot_model_comparison.png")
