import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score
from xgboost import XGBClassifier
from data_utils import prepare_data

repo_path = Path(__file__).parent / "reports" 

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    n_normal = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale_pos_weight = n_normal / n_fraud

    basemodel = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42, n_estimators=200)
    basemodel.fit(X_train, y_train)
    uncalibrated_scores = basemodel.predict_proba(X_test)[:, 1]

    basemodel_calibrated = CalibratedClassifierCV(basemodel, method="isotonic", cv=5)
    basemodel_calibrated.fit(X_train, y_train)
    calibrated_scores = basemodel_calibrated.predict_proba(X_test)[:, 1]

    brier_uncalibrated = brier_score_loss(y_test, uncalibrated_scores)
    brier_calibrated = brier_score_loss(y_test, calibrated_scores)
    print(f"Brier score, uncalibrated XGBoost: {brier_uncalibrated:.5f}")
    print(f"Brier score, calibrated XGBoost:   {brier_calibrated:.5f}")
    print("(lower = better calibration and ranking quality)")

    ap_uncalibrated = average_precision_score(y_test, uncalibrated_scores)
    ap_calibrated = average_precision_score(y_test, calibrated_scores)
    print(f"PR-AUC, uncalibrated: {ap_uncalibrated:.4f}")
    print(f"PR-AUC, calibrated:   {ap_calibrated:.4f}")


    near_zero_uncal = (uncalibrated_scores < 1e-6).sum()
    near_zero_cal = (calibrated_scores < 1e-6).sum()
    print(f"\nTransactions scored as essentially exactly 0.0:")
    print(f"  Uncalibrated: {near_zero_uncal} / {len(uncalibrated_scores)}")
    print(f"  Calibrated:   {near_zero_cal} / {len(calibrated_scores)}")
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(uncalibrated_scores, bins=50, color="C0", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Uncalibrated XGBoost - predicted probability distribution")
    axes[0].set_xlabel("Predicted probability of fraud")
    axes[0].set_ylabel("Number of transactions (log scale)")
 
    axes[1].hist(calibrated_scores, bins=50, color="C1", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_title("Calibrated XGBoost - predicted probability distribution")
    axes[1].set_xlabel("Predicted probability of fraud")
    axes[1].set_ylabel("Number of transactions (log scale)")
 
    plt.tight_layout()
    histogram_path = repo_path / "calibration_score_distribution.png"
    repo_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(histogram_path, dpi=120, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    axes[0].hist(uncalibrated_scores, bins=50, color="C0", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Uncalibrated XGBoost - predicted probability distribution")
    axes[0].set_xlabel("Predicted probability of fraud")
    axes[0].set_ylabel("Number of transactions (log scale)")
 
    axes[1].hist(calibrated_scores, bins=50, color="C1", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_title("Calibrated XGBoost - predicted probability distribution")
    axes[1].set_xlabel("Predicted probability of fraud")
    axes[1].set_ylabel("Number of transactions (log scale)")
 
    plt.tight_layout()
    histogram_path = repo_path / "calibration_score_distribution.png"
    plt.savefig(histogram_path, dpi=120, bbox_inches="tight")
    print(f"Saved score distribution histogram to {histogram_path}")