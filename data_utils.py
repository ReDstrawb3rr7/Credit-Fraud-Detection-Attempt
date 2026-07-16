import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

credit_card_raw_data = Path(__file__).parent / "data" / "raw" / "creditcard.csv"
def prepare_data(test_size=0.25, random_state=42):
    df = pd.read_csv(credit_card_raw_data)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    features = [column for column in df.columns if column.startswith("V")]
    features += ["Amount", "Time"]

    X = df[features]
    y = df["Class"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)