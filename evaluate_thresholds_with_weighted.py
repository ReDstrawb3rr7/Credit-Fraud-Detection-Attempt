import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, recall_score, confusion_matrix,f1_score, precision_score, average_precision_score)

credit_card_raw_data = Path(__file__).parent / "data" / "raw" / "creditcard.csv"
pr_graph_path = Path(__file__).parent / "reports" / "pr_graph.png"

def prepare_data():
    df = pd.read_csv(credit_card_raw_data)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    features = [column for column in df.columns if column.startswith("V")]

    features += ["Amount", "Time"]

    X = df[features]
    y = df["Class"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def threshold_evaluation(y_test, y_scores, threshold):
    print(f"Threshold = {threshold:.2f}")
    y_pred_at_threshold = (y_scores >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_at_threshold, zero_division=0)
    recall = recall_score(y_test, y_pred_at_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_at_threshold, zero_division=0)
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    cm = confusion_matrix(y_test, y_pred_at_threshold)
    false_alarms = cm[0][1]
    fraud_caught = cm[1][1]
    fraud_missed = cm[1][0]
    print(f"  Fraud caught: {fraud_caught}/{fraud_caught + fraud_missed}  "
          f"False alarms: {false_alarms}")
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    
    y_scores = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    avg_precision = average_precision_score(y_test, y_scores)
    print(f"Average precision (PR-AUC): {avg_precision:.4f}\n")

    plt.figure(figsize=(8,8))
    plt.plot(recalls, precisions, label=f"AP = {avg_precision:.3f}")
    plt.xlabel("Recall (fraction of fraud caught)")
    plt.ylabel("Precision (fraction of flagged transactions that are fraud)")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.1)
    pr_graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pr_graph_path, dpi=120, bbox_inches="tight")
    plt.show()

    print("Comparing thresholds:\n")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        threshold_evaluation(y_test, y_scores, t)
        print()

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores[:-1].argmax()  
    best_threshold = thresholds[best_idx]

    print(f"Threshold that maximises F1: {best_threshold:.3f}")
    threshold_evaluation(y_test, y_scores, best_threshold)