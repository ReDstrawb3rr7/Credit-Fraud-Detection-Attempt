import pandas as pd
from pathlib import Path

credit_card_raw_data = Path(__file__).parent / "data" / "raw" / "creditcard.csv"

if __name__ == "__main__":
    df = pd.read_csv(credit_card_raw_data)

    print(f"Shape: {df.shape}\n")

    print("Missing values per column:")
    print(df.isnull().sum().sum(), "total missing values\n")

    print("Class balance:")
    counts = df["Class"].value_counts()
    print(counts)
    fraud_pct = df["Class"].mean() * 100

    print(f"Fraud is {fraud_pct:.4f}% of all transactions\n")
    print("Amount statistics by class:")
    print(df.groupby("Class")["Amount"].describe())
    print()
    
    print("Time statistics (Time = seconds since first transaction in dataset):")
    print(f"  Range: {df['Time'].min()} to {df['Time'].max()} seconds")
    print(f"  That's about {df['Time'].max() / 3600:.1f} hours of data\n")
    print("Median transaction amount:")
    print(f"  Normal: {df[df['Class']==0]['Amount'].median():.2f}")
    print(f"  Fraud:  {df[df['Class']==1]['Amount'].median():.2f}")
