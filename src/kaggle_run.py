import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "BT2024158"

CONFIG = {
    1: {"degree": 5, "model": lambda: Lasso(alpha=0.01, max_iter=50000)},
    2: {"degree": 11, "model": lambda: Ridge(alpha=1.0)},
}

ON_KAGGLE = os.path.isdir("/kaggle/input")
SEARCH_ROOT = "/kaggle/input" if ON_KAGGLE else "."
OUT_DIR = "/kaggle/working" if ON_KAGGLE else "predictions"


ALL_CSVS = sorted(glob.glob(os.path.join(SEARCH_ROOT, "**", "*.csv"), recursive=True))


def find(kind, var):
    key = f"{kind}_var{var}"
    hits = [p for p in ALL_CSVS if key in os.path.basename(p).lower()]
    exact = [p for p in hits if ROLL.lower() in os.path.basename(p).lower()]
    if exact or hits:
        return (exact or hits)[0]
    listing = "\n  ".join(ALL_CSVS) or "(no CSV files at all - is the dataset attached via 'Add Input'?)"
    raise FileNotFoundError(f"No file containing '{key}' under {SEARCH_ROOT}. CSVs found:\n  {listing}")


os.makedirs(OUT_DIR, exist_ok=True)
for var, cfg in CONFIG.items():
    train_path, test_path = find("train", var), find("test", var)
    print(f"var{var}: train={train_path}  test={test_path}")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    feats = [c for c in train.columns if c.lower().startswith("x")]

    model = make_pipeline(
        PolynomialFeatures(degree=cfg["degree"], include_bias=False),
        StandardScaler(),
        cfg["model"](),
    )
    model.fit(train[feats].to_numpy(float), train["y"].to_numpy(float))
    pred = model.predict(test[feats].to_numpy(float))

    out_path = os.path.join(OUT_DIR, f"{ROLL}_pred_var{var}.csv")
    pd.DataFrame({"y": pred}).to_csv(out_path, index=False)
    msg = f"var{var}: degree {cfg['degree']}, {len(pred)} predictions -> {out_path}"

    if "y" in test.columns:
        y = test["y"].to_numpy(float)
        mse = np.mean((y - pred) ** 2)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        msg += f" | test MSE={mse:.4f}  R2={r2:.4f}"
    print(msg)