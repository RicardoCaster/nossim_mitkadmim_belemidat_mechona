🚍 Predicting Public Passenger Vehicle Inspection Outcomes

Course Project – Data Mining
Course: Nossim Mitkadmim Belemidat Mechona
University: Ariel University
Student: Ricardo Caster

📌 Project Overview

This project applies supervised and unsupervised machine learning techniques to predict the outcome of public passenger vehicle inspections (PASS / FAIL) using real data from data.gov.

The goal is not only to build accurate models, but also to:

Understand which factors truly influence inspection outcomes

Avoid non-causal or misleading predictors

Compare different modeling strategies in a principled way

🎯 Problem Definition

Given historical inspection records, we aim to predict whether a vehicle will:

PASS (0)

FAIL (1)

Special care is taken to:

Handle missing data correctly

Address class imbalance

Evaluate models using appropriate metrics (not accuracy alone)

📊 Dataset

Source: data.gov

Source link: https://catalog.data.gov/dataset/public-passenger-vehicle-inspection-schedule

Records: ~35,000 inspections

After cleaning: 25,567 samples

Target variable: Result → binary target

Class distribution:

~66% Passed

~34% Failed

Key Features Used

Public Vehicle Type

Vehicle Make

Vehicle Model Year → Vehicle Age

Company Name

Vehicle Status

Inspection Type

Affiliation (with missing handled explicitly)

License Management (with missing handled explicitly)

Features Explicitly Evaluated but Excluded

Inspection date, month, hour

These were tested via an ablation study and shown to add no meaningful predictive value.

🧠 Methodology
1️⃣ Data Cleaning & Feature Engineering

Filtered only PASSED / FAILED inspections

Converted target to binary classification

Engineered Vehicle Age

Treated missing categorical values as a separate category ("missing")

This preserves information and avoids bias.

2️⃣ Preprocessing Pipeline

A unified pipeline was used for all models:

Numerical features → imputation + scaling

Categorical features → imputation + one-hot encoding

This ensures:

No data leakage

Fair comparison between models

3️⃣ Supervised Learning Models

Five classification models were trained and evaluated:

Model	Purpose
Logistic Regression	Interpretable baseline
Random Forest	Non-linear ensemble
XGBoost	High-performance boosting
Support Vector Machine (SVM)	High recall for failures
K-Nearest Neighbors (KNN)	Distance-based baseline
Evaluation Metrics

Accuracy – Overall correctness

Precision – Reliability of failure predictions

Recall – Ability to detect failures (most important)

F1-Score – Precision/Recall balance

ROC-AUC – Ranking quality

PR-AUC – Performance on imbalanced data

4️⃣ Unsupervised Learning
🔹 K-Means Clustering

Used to group vehicles into latent risk profiles and analyze:

Average failure rates per cluster

Structural patterns in the data

🔹 Isolation Forest

Used for anomaly detection:

Identifies unusual vehicles

Demonstrates that anomalies ≠ failures

5️⃣ Ablation Study (Time Features)

A dedicated comparison tested models:

With inspection timing features

Without inspection timing features

Result:
Timing features do not meaningfully improve ROC-AUC.

📌 Conclusion:
Time features are excluded from the final model to preserve causality and generalization.

🏆 Key Results (Summary)

Best Recall: SVM (~79%)

Best Accuracy: XGBoost (~71%)

Best Balance: Logistic Regression / SVM

Unsupervised insights: Meaningful clusters, realistic anomaly rate (~2%)

There is no single “best” model — the choice depends on risk tolerance.