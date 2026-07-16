import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from data_utils import prepare_data

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=["Normal", "Fraud"]))
   
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    fraud_caught = cm[1][1]
    fraud_missed = cm[1][0]
    print(f"Fraud caught: {fraud_caught}/{fraud_caught + fraud_missed}  "
          f"(missed {fraud_missed})")
    print(f"False positives (normal flagged as fraud): {cm[0][1]}")

    return cm

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train size: {len(X_train)} (including {y_train.sum()} fraud)")
    print(f"Test size: {len(X_test)} (including {y_test.sum()} fraud)")

    print("\n1st approach: class weight = \"balanced\"")
    weighted_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    weighted_model.fit(X_train, y_train)
    print("\nWeighted Logistic Regression")
    evaluate_model(weighted_model, X_test, y_test)


    print("\n2nd approach: SMOTE oversampling")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"\ntraining set fraud count before SMOTE: "
          f"{y_train.sum()} \nafter SMOTE applied: {y_train_smote.sum()} "
          f"(synthetic examples added)")
    
    SMOTE_model = LogisticRegression(max_iter=1000)
    SMOTE_model.fit(X_train_smote, y_train_smote)
    print("\nSMOTE")
    evaluate_model(SMOTE_model, X_test, y_test)
    