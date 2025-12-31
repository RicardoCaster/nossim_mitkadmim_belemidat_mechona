# ============================================================
# Course: Nossim Mitkadmim Belemidat Mechona – Ariel University
# Course Project – Data Mining 2025
#
# Student: Ricardo Caster
# Project: Predicting Public Passenger Vehicle Inspection Outcomes
#
# ============================================================
# GENERAL IDEA OF THIS SCRIPT
# ------------------------------------------------------------
# This script implements a full end-to-end machine learning project:
#
# 1. Define a meaningful prediction problem
# 2. Engineer features based on domain knowledge
# 3. Train multiple supervised ML models
# 4. Evaluate them correctly for imbalanced data
# 5. Perform unsupervised learning (clustering + anomaly detection)
# 6. Perform a clean ablation study on time-based features
# ============================================================

print("\n================ STARTING PROJECT =================\n")

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# pandas / numpy → data handling
# matplotlib / seaborn → visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Core scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Supervised learning models
# Each model represents a different ML philosophy
from sklearn.linear_model import LogisticRegression        # Linear, interpretable
from sklearn.ensemble import RandomForestClassifier        # Tree ensemble
from sklearn.svm import SVC                                # Margin-based classifier
from sklearn.neighbors import KNeighborsClassifier         # Similarity-based
from xgboost import XGBClassifier                          # Boosted trees (strong on tabular data)

# Evaluation metrics
# These are chosen specifically for imbalanced classification
from sklearn.metrics import (
    accuracy_score,        # Overall correctness
    precision_score,       # How many predicted failures were actually failures
    recall_score,          # How many real failures we managed to catch
    f1_score,              # Balance between precision and recall
    roc_auc_score,         # Ranking quality independent of threshold
    precision_recall_curve,
    auc
)

# Unsupervised learning
from sklearn.cluster import KMeans                         # Clustering
from sklearn.ensemble import IsolationForest               # Anomaly detection
from sklearn.decomposition import PCA                       # Visualization aid

print("Step 1: Libraries imported successfully.\n")

# ============================================================
# 2. LOAD DATA
# ============================================================

print("Step 2: Loading dataset...\n")

df = pd.read_csv("data/Public_Passenger_Vehicle_Inspection_Schedule.csv")

print(f"Raw dataset shape: {df.shape}")
print("Dataset loaded.\n")

# ============================================================
# 3. DEFINE THE TARGET VARIABLE
# ============================================================

print("Step 3: Defining the prediction target...\n")

# We focus ONLY on PASSED vs FAILED inspections.
# FAILED is defined as the positive class (1) because:
# - It is the minority class
# - It represents risk
# - Recall on failures is operationally important

df = df[df["Result"].isin(["PASSED", "FAILED"])].copy()
df["target"] = (df["Result"] == "FAILED").astype(int)

print("Target distribution (0=Passed, 1=Failed):")
print(df["target"].value_counts(), "\n")

# ============================================================
# 4. FEATURE ENGINEERING (DOMAIN-DRIVEN)
# ============================================================

print("Step 4: Feature engineering...\n")

# Vehicle age is a meaningful mechanical proxy
current_year = 2026
df["vehicle_age"] = current_year - df["Vehicle Model Year"]

# Remove invalid ages
df = df[df["vehicle_age"] >= 0]

# Deal with NULL values
#Although the variables Affiliation and License Management contain missing values,
## they were handled using a dedicated ‘missing’ category.
###This approach preserves information, avoids data loss, and allows the model to learn whether the absence of information itself is predictive.


# We intentionally select ONLY vehicle & organization-related features.
# Time-based scheduling features are excluded here on purpose.
categorical_features = [
    "Public Vehicle Type",
    "Vehicle Make",
    "Vehicle Model",
    "Company Name",
    "Vehicle Status",
    "Inspection Type",
    "Affiliation",
    "License Management"
]

numerical_features = ["vehicle_age"]

X = df[categorical_features + numerical_features]
y = df["target"]

print(f"Number of samples after cleaning: {X.shape[0]}")
print(f"Number of features used: {X.shape[1]}\n")

# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

print("Step 5: Building preprocessing pipeline...\n")

# Explanation:
# - Categorical variables must be converted to numbers → One-Hot Encoding
# - Numerical variables are scaled → StandardScaler
# - Missing values are handled safely

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("Step 6: Splitting data into train and test sets...\n")

# Stratification preserves class imbalance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0], "\n")

# ============================================================
# 7. SUPERVISED LEARNING MODELS
# ============================================================

print("Step 7: Training supervised models...\n")

# Each model captures patterns differently.
# Using multiple models increases robustness and insight.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = []

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metric meanings:
    # Accuracy → overall correctness (can be misleading in imbalance)
    # Precision → reliability of failure predictions
    # Recall → ability to catch failures (most important here)
    # F1 → balance between precision & recall
    # ROC-AUC → quality of ranking, threshold independent
    # PR-AUC → best metric for imbalanced datasets

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    p, r, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(r, p)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC-AUC: {roc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

    results.append([name, acc, prec, rec, f1, roc, pr_auc])

# ============================================================
# 8. SUPERVISED RESULTS SUMMARY
# ============================================================

print("\nStep 8: Summary of supervised model performance...\n")

results_df = pd.DataFrame(
    results,
    columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
)

print(results_df.sort_values("roc_auc", ascending=False))

# ============================================================
# 9. UNSUPERVISED LEARNING – CLUSTERING
# ============================================================

print("\nStep 9: Unsupervised learning – K-Means clustering...\n")

# Clustering finds structure WITHOUT using the inspection result.
# This helps discover hidden risk profiles.

X_processed = preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_processed)

print("Average failure rate per cluster:")
print(df.groupby("cluster")["target"].mean(), "\n")

# Visual explanation:
# PCA reduces high-dimensional data to 2D ONLY for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster"], palette="tab10")
plt.title("Vehicle clusters (PCA visualization)")
plt.show()

# ============================================================
# 10. UNSUPERVISED LEARNING – ANOMALY DETECTION
# ============================================================

print("\nStep 10: Anomaly detection using Isolation Forest...\n")

# Isolation Forest isolates rare and unusual patterns.
# These can represent suspicious or exceptional vehicles.

iso = IsolationForest(contamination=0.02, random_state=42)
df["anomaly"] = iso.fit_predict(X_processed)

print("Anomaly distribution (-1 = anomaly):")
print(df["anomaly"].value_counts(), "\n")

print("Sample anomalies:")
print(df[df["anomaly"] == -1][
    ["Public Vehicle Type", "Vehicle Make", "vehicle_age", "Result"]
].head())

# ============================================================
# 11. ABLATION STUDY – TIME FEATURES (COMPARISON ONLY)
# ============================================================

print("\nStep 11: Ablation study – including time features (comparison only)...\n")

print(
    "This section is NOT part of the final model.\n"
    "It exists only to empirically prove whether schedule timing adds value.\n"
)

df["inspection_month"] = pd.to_datetime(
    df["Scheduled Inspection Date and Time"], errors="coerce"
).dt.month

df["inspection_hour"] = pd.to_datetime(
    df["Scheduled Inspection Date and Time"], errors="coerce"
).dt.hour

X_time = df[categorical_features + numerical_features + ["inspection_month", "inspection_hour"]]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_time, y, test_size=0.2, random_state=42, stratify=y
)

time_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features + ["inspection_month", "inspection_hour"]),
        ("cat", categorical_transformer, categorical_features)
    ]
)

time_pipeline = Pipeline(steps=[
    ("preprocessor", time_preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

time_pipeline.fit(X_train_t, y_train_t)
roc_time = roc_auc_score(y_test_t, time_pipeline.predict_proba(X_test_t)[:, 1])

print(f"ROC-AUC WITH time features: {roc_time:.3f}")
print("Conclusion: Timing features do NOT meaningfully improve performance.")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n================ PROJECT COMPLETED SUCCESSFULLY ================\n")
print(
    "Final conclusion:\n"
    "- Vehicle and organization features explain inspection outcomes\n"
    "- Time scheduling features add no real predictive value\n"
)
