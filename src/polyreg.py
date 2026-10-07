import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ALPHAS = {
    "ridge": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    "lasso": [1e-4, 3e-4, 1e-3, 3e-3, 0.007, 0.01, 0.015, 0.02, 0.03, 0.1],
}


def build_model(degree, alpha, penalty="ridge"):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha) if penalty == "ridge" else Lasso(alpha=alpha, max_iter=50000),
    )


def feature_columns(df):
    return [c for c in df.columns if c.lower().startswith("x")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roll", required=True)
    p.add_argument("--var", required=True, type=int)
    p.add_argument("--max-degree", type=int, default=10)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--penalty", choices=["ridge", "lasso"], default="ridge")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(args.data_dir) / f"{args.roll}_train_var{args.var}.csv")
    feats = feature_columns(train)
    X, y = train[feats].to_numpy(float), train["y"].to_numpy(float)
    print(f"var{args.var}: train {X.shape}, features {feats}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    rows = []
    for degree in range(1, args.max_degree + 1):
        n_terms = PolynomialFeatures(degree, include_bias=False).fit(X[:1]).n_output_features_
        for alpha in ALPHAS[args.penalty]:
            pred = cross_val_predict(build_model(degree, alpha, args.penalty), X, y, cv=kf, n_jobs=-1)
            rows.append({"degree": degree, "alpha": alpha, "n_terms": n_terms,
                         "cv_mse": mean_squared_error(y, pred), "cv_r2": r2_score(y, pred)})
        best_d = min((r for r in rows if r["degree"] == degree), key=lambda r: r["cv_mse"])
        print(f"  degree {degree:2d} ({n_terms:5d} terms): best alpha={best_d['alpha']:g} "
              f"CV MSE={best_d['cv_mse']:.6g}  CV R2={best_d['cv_r2']:.6f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / f"cv_results_var{args.var}.csv", index=False)
    best = results.loc[results["cv_mse"].idxmin()]
    degree, alpha = int(best["degree"]), float(best["alpha"])
    print(f"Selected degree={degree}, alpha={alpha:g}, "
          f"CV MSE={best['cv_mse']:.6g}, CV R2={best['cv_r2']:.6f}")

    model = build_model(degree, alpha, args.penalty).fit(X, y)
    train_pred = model.predict(X)
    summary = {
        "var": args.var, "features": feats, "penalty": args.penalty,
        "degree": degree, "alpha": alpha,
        "nonzero_coefs": int(np.sum(model[-1].coef_ != 0)),
        "n_terms": int(best["n_terms"]), "folds": args.folds,
        "cv_mse": float(best["cv_mse"]), "cv_r2": float(best["cv_r2"]),
        "train_mse": float(mean_squared_error(y, train_pred)),
        "train_r2": float(r2_score(y, train_pred)),
        "n_train": len(y), "y_var": float(np.var(y)),
    }
    (out_dir / f"summary_var{args.var}.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / f"model_var{args.var}.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feats}, f)
    print(f"Saved {out_dir / f'model_var{args.var}.pkl'}")


if __name__ == "__main__":
    main()