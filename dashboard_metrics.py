"""
Creates dashboard_metrics.pkl WITHOUT retraining anything.
Run from your project folder (venv active):   python make_dashboard_metrics.py

Needs these files next to it (all already on your computer):
  credit_risk_dataset.csv, logistic_model.pkl, xgb_model.pkl, rf_model.pkl, thresholds.pkl
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss, precision_score,
                             recall_score, f1_score)

# ---- same cleaning and split as the notebook, so the test set is identical
df = pd.read_csv("credit_risk_dataset.csv").drop_duplicates()
df = df[df["person_age"].between(18, 100)]
df = df[df["person_emp_length"].isna() |
        ((df["person_emp_length"] <= df["person_age"]) & (df["person_emp_length"] <= 60))]
df = df[(df["person_income"] > 0) & (df["loan_amnt"] > 0) & (df["person_income"] <= 1_000_000)]
df = df[df["loan_int_rate"].isna() | df["loan_int_rate"].between(0, 40)]
df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
X, y = df.drop("loan_status", axis=1), df["loan_status"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- load the three saved models and their thresholds
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "XGBoost (calibrated)": "xgb_model.pkl",
    "Random Forest (calibrated)": "rf_model.pkl",
}
thresholds = joblib.load("thresholds.pkl")

perf, rows = {}, []
for name, file in MODEL_FILES.items():
    print("Evaluating", name, "...")
    model = joblib.load(file)
    thr = float(thresholds[name])
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_test, prob)
    prec, rec, _ = precision_recall_curve(y_test, prob)
    frac, mean = calibration_curve(y_test, prob, n_bins=10)
    perf[name] = dict(threshold=thr, fpr=fpr, tpr=tpr, precision=prec, recall=rec,
                      frac_pos=frac, mean_pred=mean, cm=confusion_matrix(y_test, pred))
    rows.append({"Model": name,
                 "ROC-AUC": roc_auc_score(y_test, prob),
                 "PR-AUC": average_precision_score(y_test, prob),
                 "Brier (lower=better)": brier_score_loss(y_test, prob),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "Recall": recall_score(y_test, pred, zero_division=0),
                 "F1": f1_score(y_test, pred, zero_division=0),
                 "Threshold": thr})
    del model

comparison = pd.DataFrame(rows).set_index("Model").round(4)
champion_name = comparison["PR-AUC"].idxmax()          # same rule as the notebook

joblib.dump(dict(comparison=comparison, perf=perf, champion_name=champion_name), "dashboard_metrics.pkl")

print()
print(comparison.to_string())
print("\nChampion:", champion_name)
print("Saved dashboard_metrics.pkl")
