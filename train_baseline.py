import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)

credit_card_raw_data = Path(__file__).parent / "data" / "raw" / "creditcard.csv"

if __name__ == "__main__":
    df = pd.read_csv(credit_card_raw_data)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    features = [column for column in df.columns if column.startswith("V")]

    features += ["Amount", "Time"]

    X = df[features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # print(X, "\n")
    # print("next...............")
    # print(y, "\n")
    print("Training examples:", len(X_train))
    print("Fraud examples in training:", y_train.sum())

    print("Testing examples:", len(X_test))
    print("Fraud examples in testing:", y_test.sum())

    #Logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    #predictions
    predictions = model.predict(X_test)
    #evaluate model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Fraud"]
        )
    )

    #fraud detection
    matrix = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print("                     Predicted Normal    Predicted Fraud")
    print(f"Actual Normal      {matrix[0][0]}          {matrix[0][1]}")
    print(f"Actual Fraud       {matrix[1][0]}          {matrix[1][1]}")

    fraud_found = matrix[1][1]
    fraud_missed = matrix[1][0]

    print(f"\nOut of {fraud_found + fraud_missed} fraud cases:")
    print(f"Caught: {fraud_found}")
    print(f"Missed: {fraud_missed}")