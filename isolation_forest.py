import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

credit_card_raw_data = Path(__file__).parent / "data" / "raw" / "creditcard.csv"

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

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    actual_fraud_rate = y_train.mean()
    print(f"Train size: {len(X_train)} (including {y_train.sum()} fraud)")
    print(f"{actual_fraud_rate:.4f} of training examples are fraud (not truly blind setup)\n")

    iso_forest = IsolationForest(contamination=actual_fraud_rate, random_state=42, n_estimators=100)
    iso_forest.fit(X_train)

    raw_preds = iso_forest.predict(X_test)
    y_pred = pd.Series(raw_preds).map({1: 0, -1: 1}).values

    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: predicted_normal  predicted_fraud")
    print(f"actual_normal      {cm[0][0]}          {cm[0][1]}")
    print(f"actual_fraud       {cm[1][0]}             {cm[1][1]}")
    fraud_caught = cm[1][1]
    fraud_missed = cm[1][0]
    
    print(f"\nFraud caught: {fraud_caught}/{fraud_caught + fraud_missed}  "
          f"(missed {fraud_missed})")
    print(f"False positives: {cm[0][1]}")