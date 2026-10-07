# Polynomial Regression Assignment (BT2024158)

Polynomial regression models for two personalised datasets. In both, the polynomial degree and the
regularisation strength are chosen by 5-fold cross-validation on the training set.

| Problem | Inputs | Model | Degree | alpha | CV MSE | CV R² |
|---|---|---|---|---|---|---|
| var1: steam turbine Net Power Score | x1–x6 | Lasso polynomial | 5 | 0.01 | 0.353 | 0.963 |
| var2: thermal anomaly score | x1–x3 | Ridge polynomial | 11 | 1.0 | 0.250 | 0.995 |

See [`report/report.pdf`](report/report.pdf) for the approach, the CV results at every degree, and the reasoning behind each choice.

## Repository structure

```
data/           training and test CSVs, sample_submission.csv
src/
  polyreg.py    grid search over (degree, alpha) with 5-fold CV; saves model + CV table + summary
  predict.py    loads a saved model and writes predictions/<ROLLNO>_pred_var<ID>.csv
  kaggle_run.py self-contained train + predict script for a Kaggle notebook
outputs/        saved models (model_var*.pkl), full CV results, summaries, training logs
  var1_ridge_comparison/  Ridge results for var1 (compared against Lasso in the report)
predictions/    BT2024158_pred_var1.csv, BT2024158_pred_var2.csv (single column `y`)
report/         report.tex and report.pdf
```

## Setup

```bash
pip install -r requirements.txt
```

## Train (from the repo root)

```bash
python src/polyreg.py --roll BT2024158 --var 1 --max-degree 8  --penalty lasso
python src/polyreg.py --roll BT2024158 --var 2 --max-degree 20 --penalty ridge
# Ridge comparison for var1 (Table 2 in the report)
python src/polyreg.py --roll BT2024158 --var 1 --max-degree 10 --penalty ridge --out-dir outputs/var1_ridge_comparison
```

## Inference

```bash
python src/predict.py --roll BT2024158 --var 1
python src/predict.py --roll BT2024158 --var 2
```

This writes `predictions/BT2024158_pred_var1.csv` and `predictions/BT2024158_pred_var2.csv`.

### On Kaggle

Attach the train and test CSVs as notebook input, paste `src/kaggle_run.py` into a cell and run it.
It finds the files anywhere under `/kaggle/input` and writes both prediction files to `/kaggle/working`.
