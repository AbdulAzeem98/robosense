"""
RoboSense: Feature extraction for force/torque time-series windows.

Turns each (15, 6) raw instance into a flat feature vector so classic
models (XGBoost, Logistic Regression) can use it, while the raw
sequence is kept separately for the LSTM.
"""
import numpy as np
import pandas as pd

AXIS_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def extract_features(X: np.ndarray) -> pd.DataFrame:
    """X: (n_instances, 15, 6) -> DataFrame of statistical + FFT features."""
    rows = []
    for instance in X:
        feats = {}
        for i, axis in enumerate(AXIS_NAMES):
            series = instance[:, i]
            feats[f"{axis}_mean"] = series.mean()
            feats[f"{axis}_std"] = series.std()
            feats[f"{axis}_min"] = series.min()
            feats[f"{axis}_max"] = series.max()
            feats[f"{axis}_range"] = series.max() - series.min()
            # skew (simple, no scipy dependency)
            std = series.std()
            feats[f"{axis}_skew"] = (
                ((series - series.mean()) ** 3).mean() / (std ** 3) if std > 1e-9 else 0.0
            )
            # dominant FFT frequency magnitude (captures oscillation/vibration)
            fft_vals = np.abs(np.fft.rfft(series - series.mean()))
            feats[f"{axis}_fft_energy"] = (fft_vals ** 2).sum()
            feats[f"{axis}_fft_dom"] = fft_vals[1:].max() if len(fft_vals) > 1 else 0.0
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_dataset
    X, y = load_dataset("data/lp1.data.txt")
    df = extract_features(X)
    print(df.shape)
    print(df.head())
