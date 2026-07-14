
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

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

    # A quick, honest gut-check: are fraud transactions typically smaller
    # or larger amounts? This kind of basic look at the data often
    # reveals patterns worth keeping in mind before modelling.
    print("Median transaction amount:")
    print(f"  Normal: {df[df['Class']==0]['Amount'].median():.2f}")
    print(f"  Fraud:  {df[df['Class']==1]['Amount'].median():.2f}")
