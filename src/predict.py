import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roll", required=True)
    p.add_argument("--var", required=True, type=int)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="outputs")
    p.add_argument("--pred-dir", default="predictions")
    args = p.parse_args()

    with open(Path(args.model_dir) / f"model_var{args.var}.pkl", "rb") as f:
        saved = pickle.load(f)

    test = pd.read_csv(Path(args.data_dir) / f"{args.roll}_test_var{args.var}.csv")
    pred = saved["model"].predict(test[saved["features"]].to_numpy(float))

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{args.roll}_pred_var{args.var}.csv"
    pd.DataFrame({"y": pred}).to_csv(out_path, index=False)
    print(f"Wrote {len(pred)} predictions to {out_path}")


if __name__ == "__main__":
    main()