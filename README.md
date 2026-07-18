# Credit Card Fraud Detection

Exploring imbalanced classification using a real anonymized credit card transaction dataset. This is my second project. I picked it specifically because my first one (a stock sentiment classifier) had a mild class imbalance that quietly broke the model, and I wanted to properly learn how to deal with that instead of just noting it as a limitation and moving on.

## TL;dr

- Dataset: 284,807 real transactions, only 492 (0.17%) fraudulent. Extreme imbalance.
- A plain Logistic Regression baseline got 99.92% accuracy while missing over a third of actual fraud
- Tried class weighting and SMOTE to fix recall - both worked (63% to 89% fraud caught), but at the cost of precision collapsing (0.85 to 0.06, meaning ~1700+ false alarms).
- Tried Isolation Forest (unsupervised anomaly detection) as a genuinely different approach. It performed worse than the plain baseline, which taught me more about when unsupervised methods actually earn their keep than a "win" would have.
- Tuned the decision threshold properly using precision-recall curves instead of accepting the default 0.5 cutoff. Found a much better balance (96/123 caught, only 16 false alarms).
- Tried XGBoost, since this dataset (284k rows) is actually big enough to justify a more complex model, unlike my first project's 86-row dataset. It beat every prior attempt: 98/123 caught with only 12 false alarms by default, and 97/123 with just 7 false alarms once threshold-tuned.
- Added probability calibration on top of XGBoost. The standard reliability diagram barely showed a difference (bin-averaging hides the effect when data's this imbalanced), so I dug deeper and found the real story: 75% of transactions were getting an identical, exact 0.0 score from the raw model, versus 0% after calibration. A genuinely more informative probability spread, not just a cosmetic fix.
- Refactored repeated data-loading code into a shared module partway through, make code cleaner and easier to read.

## Dataset

[Credit Card Fraud Detection - Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

284,807 real, anonymized European transactions, 492 of them fraudulent. Most of the features (V1 through V28) are the output of PCA, done by whoever compiled the dataset to protect the actual transaction details, so I don't know what any individual feature actually represents. Time and Amount are the only two left in their original form. 

## Before Modelling

Fraud transactions have a lower median amount than normal ones ($9.25 vs $22), but a higher mean ($122 vs $88). My read on that is most fraud attempts are small, maybe testing whether stolen card details work before trying something bigger, but there are a few large fraud outliers pulling the average up. I don't know if that's actually the right explanation, but it was interesting enough to note before I started modelling.

## Baseline

I trained a plain Logistic Regression with no special handling for the imbalance, just to retain simplicity before doing anything else. It got 99.92% accuracy, which was quite worrying. This concern was affirmed when the model only caught 77 out of 123 fraud cases in the test set. Missing over a third of actual fraud while still posting a 99.9% accuracy score is exactly the kind of thing that makes accuracy a bad metric to lean on here. Precision was decent (0.85, so it wasn't raising too many false alarms), but recall was the real problem.

## Fixing Recall: class weighting and SMOTE

I tried two different approaches. Class weighting, which just tells the model to penalise mistakes on fraud more heavily without creating any new data, and SMOTE, which generates synthetic fraud examples by interpolating between real ones in feature space (only applied to the training set, since applying it before the split would leak information from the test set).

Both pushed recall up a lot, from 77/123 caught to 109/123. But precision fell off a cliff, from 0.85 down to about 0.06. That means around 1,600-1,700 legitimate transactions got flagged as fraud in exchange for catching those extra 32 real fraud cases. Interestingly, class weighting and SMOTE landed at almost identical numbers, so neither one was clearly better than the other here.

I think this was the most useful thing I learned early on in this project: there's a real trade-off between catching more fraud and flagging fewer innocent transactions, and there isn't an objectively correct answer. It depends on what a false alarm actually costs a business versus what missed fraud costs, which isn't something you can solve purely with better modelling.

## Isolation Forest

Everything up to this point was supervised. The model learns directly from labelled fraud examples. I wanted to try something fundamentally different, so I tried Isolation Forest, an anomaly detection method that isolates points using random decision trees. Rare, unusual points tend to get separated from the rest of the data in fewer random splits, so the average number of splits needed becomes an anomaly score. It doesn't use labels to train at all, only to check its performance afterward.

It caught only 29 out of 123 fraud cases, worse than even the plain baseline. Honestly, in hindsight this makes sense: I was throwing away information I actually had. We know which transactions are fraud in training data, and a supervised model gets to use that directly. Isolation Forest ignores it and just asks "does this look statistically unusual," which is a harder, blinder task. The lesson here isn't that Isolation Forest is a bad algorithm, it's that it's solving a more general problem than I actually needed solved, given that I had good labels available. It would likely matter a lot more in a setting where labelled fraud data is scarce or fraud patterns are genuinely novel.

## Tuning the decision threshold

Every model so far was using a default 0.5 cutoff to decide fraud vs not fraud, which is an arbitrary line that has nothing to do with the actual problem. I looked at the full precision-recall curve instead of a single cutoff, and compared the trade-off across a whole range of thresholds directly (loose thresholds catch more fraud but drown in false alarms, strict ones do the opposite).

Picking the threshold that maximises F1 score gave 96 out of 123 fraud caught with only 16 false alarms, a noticeably better balance than either of the earlier all-or-nothing attempts. One thing worth flagging honestly: the best threshold landed almost at 1.0, which hinted the class-weighted model's probability outputs were skewed toward the extremes rather than being smoothly spread out, this came back later when I looked at calibration.

## XGBoost

Every attempt so far kept hitting a similar ceiling with Logistic Regression, so I wanted to try a stronger model. In my first project, I deliberately avoided anything more complex than Logistic Regression because I only had 86 rows to work with, and a complex model would have just overfit. Here I have 284,807 rows, which is a genuinely different situation, so using a more powerful model is actually justified rather than reckless.

XGBoost handles the imbalance itself through `scale_pos_weight`, which tells the boosting process to penalise mistakes on the rare class more heavily, conceptually similar to class weighting but built into how the trees are boosted rather than the loss function.

With just the default threshold, XGBoost caught 98/123 fraud cases with only 12 false alarms, already beating everything before it. PR-AUC came out to 0.850 versus Logistic Regression's 0.706, so this wasn't just a lucky threshold, it's a genuinely stronger model. After tuning the threshold the same way as before, it hit 97/123 caught with just 7 false alarms, the best result in the whole project.

## Calibrating XGBoost

I wanted to check something specific here: does XGBoost's predicted probability actually mean what it claims to mean? A model can be excellent at ranking transactions by suspicion while still being poorly calibrated, e.g. something it scores as "80% likely fraud" might not actually be fraud 80% of the time in reality. Ranking quality and calibration quality are genuinely different properties.

I wrapped the model with `CalibratedClassifierCV` using isotonic regression. Brier score (a measure of how well predicted probabilities match actual outcomes) improved from 0.00048 to 0.00039, and PR-AUC even ticked up slightly (0.850 to 0.863), which is likely a side effect of the internal cross-validation acting like light ensembling rather than calibration itself improving ranking.

I initially checked this with the standard tool for the job, a reliability diagram, which bins predictions and compares the average predicted probability against the actual outcome rate in each bin. It barely showed any difference between the calibrated and uncalibrated model. At first I assumed calibration just wasn't doing much here. But the more likely explanation, once I thought it through, is that with fraud this rare, most bins average out to "close to zero either way," which hides what's actually happening to individual predictions.

So I looked at the raw predicted scores directly instead of bin averages, and found the real story: 53,585 out of 71,202 transactions (75%) got an identical, exact 0.0 score from the uncalibrated model, versus 0 out of 71,202 after calibration. This is a known property of tree ensembles. Many transactions land in the same leaf nodes and get identical raw scores, causing clumping at the extremes. Calibration broke that clumping into a smoother, more genuinely informative spread. I ended up dropping the reliability diagram from the repo entirely once the histogram made the actual effect obvious, since keeping a plot that didn't clearly show anything didn't add value once I had one that did.

## Results summary

| Approach | Fraud caught | False alarms | PR-AUC |
|---|---|---|---|
| Baseline Logistic Regression (default threshold) | 77/123 | 14 | - |
| Class-weighted / SMOTE Logistic Regression | 109/123 | ~1,650 | - |
| Logistic Regression, tuned threshold | 96/123 | 16 | 0.706 |
| Isolation Forest (unsupervised) | 29/123 | 98 | - |
| XGBoost (default threshold) | 98/123 | 12 | 0.850 |
| XGBoost, tuned threshold | 97/123 | 7 | 0.863 (calibrated) |

## Limitations

- The train/test split is random, not time-based. In a real deployment, you'd train on past transactions and test on future ones, since fraud patterns evolve over time. A random split like the one I used can be more optimistic than what you'd actually see in production.
- Isolation Forest's `contamination` parameter and the threshold tuning steps both used the actual fraud rate/test labels, which is a bit of an unfair advantage.
- The "best F1" threshold treats catching fraud and avoiding false alarms as equally important, which is probably not true in a real business. The actual right threshold depends on the real cost of a missed fraud versus a false alarm, which I don't have numbers for here.
- V1-V28 being anonymized PCA features means I can't sanity-check why the model flags what it flags beyond looking at raw feature importances. Basically, there's no real interpretability at the feature-meaning level.
- Only 48 hours of transactions are represented in this dataset, so there's no way to check how any of this holds up over longer timeframes or across different fraud patterns.

## What to do next time 

- A proper time-based train/test split instead of a random one
- Actual cost-sensitive threshold selection, if I can find reasonable estimates for what a false alarm vs a missed fraud costs
- Hyperparameter tuning for XGBoost rather than using mostly default settings. This is my first deployment of it so gonna need more time understanding it.
- Comparing against a deep learning approach (e.g. an autoencoder trained only on normal transactions) as another unsupervised angle, now that I have a supervised ceiling to compare it against

## Setup

```bash
python -m venv venv
source venv/bin/activate
```

There are noticeably many scripts so here is the order of the scripts based on when they are built: `explore_data.py`, `train_baseline.py`, `train_balanced.py`, `train_isolation_forest.py`, `evaluate_thresholds.py`, `train_xgboost.py`, `calibrate_xgboost.py`.

