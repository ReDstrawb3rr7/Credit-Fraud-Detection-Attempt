import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_curve, average_precision_score)
from data_utils import prepare_data
from xgboost import XGBClassifier

xgboost_pr_graph_path = Path(__file__).parent / "reports" / "xgboost_pr_graph.png"

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    n_normal = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale_pos_weight = n_normal / n_fraud
    print(f"scale_pos_weight = {scale_pos_weight:.1f}\n")

    model1 = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42, n_estimators=200)
    model1.fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)
    print("default 0.5 threshold")
    print(classification_report(y_test, y_pred1, target_names=["normal", "fraud"]))

    y_scores1 = model1.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores1)
    avg_precision = average_precision_score(y_test, y_scores1)
    print(f"XGBoost average precision (PR-AUC): {avg_precision:.4f}")
    print("Logistic Regression's PR-AUC: 0.7062\n")

    plt.figure(figsize=(8,8))
    plt.plot(recalls, precisions, label=f"XGBoost, AP = {avg_precision:.3f}")
    plt.xlabel("Recall (fraction of fraud caught)")
    plt.ylabel("Precision (fraction of flagged transactions that are fraud)")
    plt.title("Precision-Recall Curve - XGBoost")
    plt.legend()
    plt.grid(True, alpha=0.1)
    xgboost_pr_graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(xgboost_pr_graph_path, dpi=120, bbox_inches="tight")
    plt.show()

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores[:-1].argmax()
    best_threshold = thresholds[best_idx]
    y_pred_best = (y_scores1 >= best_threshold).astype(int)

    print(f"\nBest F1 threshold: {best_threshold:.3f}")
    cm_best = confusion_matrix(y_test, y_pred_best)
    print(f"Fraud caught: {cm_best[1][1]}/{cm_best[1][0]+cm_best[1][1]}  "
          f"False alarms: {cm_best[0][1]}")