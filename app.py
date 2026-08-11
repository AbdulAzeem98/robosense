"""
RoboSense: Streamlit demo app.
Upload a robot sensor trace (or pick a sample) and see the predicted
failure type plus a SHAP-based explanation.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from data_loader import load_dataset
from features import extract_features
from train import train_xgboost

st.set_page_config(page_title="RoboSense", page_icon="🤖", layout="centered")
st.title("🤖 RoboSense: Robot Failure Diagnosis")
st.caption("Predictive failure diagnosis for robotic assembly systems using sensor time-series ML")


@st.cache_resource
def load_and_train(lp_path="data/lp1.data.txt"):
    X_raw, y_raw = load_dataset(lp_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    feat_df = extract_features(X_raw)
    result = train_xgboost(feat_df.values, y_enc, list(le.classes_))
    return result, le, feat_df, X_raw, y_raw


result, le, feat_df, X_raw, y_raw = load_and_train()
model = result["model"]
explainer = shap.TreeExplainer(model)

st.subheader("1. Pick a sample sensor trace")
idx = st.slider("Sample index", 0, len(X_raw) - 1, 0)
st.write(f"**Actual label:** `{y_raw[idx]}`")

fig, ax = plt.subplots(figsize=(6, 3))
for i, axis in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
    ax.plot(X_raw[idx, :, i], label=axis)
ax.set_xlabel("Timestep (21ms intervals)")
ax.set_ylabel("Sensor reading")
ax.legend(loc="upper right", fontsize=7, ncol=3)
st.pyplot(fig)

st.subheader("2. Prediction")
x_input = feat_df.values[idx : idx + 1]
pred_class_idx = model.predict(x_input)[0]
pred_proba = model.predict_proba(x_input)[0]
pred_class = le.classes_[pred_class_idx]

st.metric("Predicted failure type", pred_class)
proba_df = pd.DataFrame({"class": le.classes_, "probability": pred_proba}).sort_values(
    "probability", ascending=False
)
st.bar_chart(proba_df.set_index("class"))

st.subheader("3. Why this prediction? (SHAP explanation)")
shap_values = explainer.shap_values(x_input)
if isinstance(shap_values, list):
    sv = shap_values[pred_class_idx][0]
else:
    sv = shap_values[0, :, pred_class_idx] if shap_values.ndim == 3 else shap_values[0]

top_idx = np.argsort(np.abs(sv))[::-1][:8]
explain_df = pd.DataFrame({
    "feature": [feat_df.columns[i] for i in top_idx],
    "shap_value": [sv[i] for i in top_idx],
}).set_index("feature")
st.bar_chart(explain_df)
st.caption("Positive values push toward the predicted class; negative values push away from it.")

st.divider()
st.caption(
    f"Model: XGBoost | Macro F1 on held-out test set: {result['macro_f1']:.3f} "
    "| Dataset: UCI Robot Execution Failures (LP1)"
)
