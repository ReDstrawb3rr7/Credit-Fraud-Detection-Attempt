# credit card fraud detection
Exploring imbalanced classification using a real anonymized credit card transaction dataset.

## dataset
[Credit Card Fraud Detection - Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

It's the well-known Kaggle credit card fraud dataset - 284,807 real, anonymized European transactions, 492 of them fraudulent. Most of the features (V1 through V28) are the output of PCA, done by whoever compiled the dataset to protect the actual transaction details, so I don't know what any individual feature actually represents. Time and Amount are the only two left in their original form.

## before modelling
Fraud transactions have a lower median amount than normal ones ($9.25 vs $22), but a higher mean ($122 vs $88). My read on that is most fraud attempts are small, maybe testing whether stolen card details work before trying something bigger, but there are a few large fraud outliers pulling the average up. I don't know if that's actually the right explanation, but it was interesting enough to note before I started modelling.

## Baseline
I initially trained a plain Logistic Regression with no special handling for the imbalance, just to retain simplicity before doing anything else. It got 99.92% accuracy, which was quite worrying. THis concern was affirmed when the model only caught 77 out of 123 fraud cases in the test set. Missing over a third of actual fraud while still posting a 99.9% accuracy score is exactly the kind of thing that makes accuracy a bad metric to lean on here. Precision was decent (0.85, so it wasn't raising too many false alarms), but recall was the real problem.

## Explored Alternative Approaches
I tried two different approaches to this. Class weighting, which just tells the model to penalise mistakes on fraud more heavily without creating any new data, and SMOTE, which generates synthetic fraud examples by interpolating between real ones in feature space (only applied to the training set, since applying it before the split would leak information from the test set).

Both pushed recall up a lot, from 77/123 caught to 109/123. But precision fell off a cliff, from 0.85 down to about 0.06. That means around 1,600-1,700 legitimate transactions got flagged as fraud in exchange for catching those extra 32 real fraud cases. Interestingly, class weighting and SMOTE landed at almost identical numbers, so neither one was clearly better than the other here.

I think this is actually the most useful thing I've learned so far in this project: there's a real trade-off between catching more fraud and flagging fewer innocent transactions, and there isn't an objectively correct answer. It depends on what a false alarm actually costs a business versus what missed fraud costs, which isn't something you can solve purely with better modelling.

## setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
