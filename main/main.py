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

# This note serves as a brief introduction to the code.
# The project was developed based on the material taught in class, complemented by
# independent study using online resources. During the development process, large
# language models were used for every line of code as supportive tools (including ChatGPT and Gemini) through
# iteration and comparison, helping to improve code organization, clarity of comments,
# and the overall reasoning behind modeling choices. Final decisions, structure, and
# understanding remain the result of my own learning and implementation.


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

# pipeline for numeric features to handle missing values and scale them.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Replace missing numeric values (NaNs) with the median of the column. Median is robust to outliers (better than mean if there are extreme values).
    ("scaler", StandardScaler())  # StandardScaler transforms numeric values to have mean=0 and std=1. Some models (like SVM, KNN) are sensitive to scale.
])

# pipeline for categorical features to handle missing values and convert categories to numbers.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # Fill missing categorical values with a constant string "missing".
# One-Hot Encoding: Convert categorical variables into binary columns (0/1) for each category.
# "vehicle_type" = ["Taxi", "Bus", "Ambulance"] → becomes 3 columns: vehicle_type_Taxi, vehicle_type_Bus, vehicle_type_Ambulance (1 where it matches, 0 otherwise)        
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # handle_unknown="ignore" → if new category appears in test data, it won’t crash the model.
])

# ColumnTransformer applies different preprocessing pipelines to different columns, allowing us to handle numeric and categorical features separately in one step.
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

# Split the dataset into Training set (to train the models) and Test set (never seen during training, used only for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X,    # feature matrix
    y,    # target vector
    test_size=0.2,    # 20% of the data used for testing
    random_state=42,  # fixed seed for reproducibility
    stratify=y   # ensures that the class distribution (Passed / Failed) remains similar in both train and test sets (VERY important when the dataset is imbalanced).
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0], "\n")

# ============================================================
# 7. SUPERVISED LEARNING MODELS
# ============================================================

print("Step 7: Training supervised models...\n")

# Define multiple supervised learning models.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"), #balanced automatically gives more weight to the minority class, forcing the model to care about it. We set this because PASSED is much more common than FAILED (typical in many real datasets), models can “cheat” by predicting PASSED all the time.
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42), # probability=True enables probability outputs(needed for ROC/AUC);
    "KNN": KNeighborsClassifier(n_neighbors=7)   # number of nearest neighbors : Looks at the 7 most similar inspections, if most neighbors failed → predict FAILED
}

results = []   # This list will store evaluation results for all models

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)   # Train the model using the training data

    y_pred = pipeline.predict(X_test)   # Predict final class labels (0 or 1)
    y_prob = pipeline.predict_proba(X_test)[:, 1]   # Predict probabilities for the positive class (Failed = 1) (needed for ROC-AUC and PR-AUC)

    # Model evaluation metrics
    acc = accuracy_score(y_test, y_pred)   # How often am I correct overall?
    prec = precision_score(y_test, y_pred) # When I predict FAILED, how often am I right? (few false alarms, important if inspections are expensive)
    rec = recall_score(y_test, y_pred)     # Of all real FAILED inspections, how many did I catch? (most important here - few missed failures)
    f1 = f1_score(y_test, y_pred)          # Balance between precision and recall (Useful when you need both: not too many false alarms AND not too many missed failures)
    roc = roc_auc_score(y_test, y_prob)    # How well the model ranks failures above passes across all thresholds (shows trade-off between catching failures and raising false alarms)

    p, r, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(r, p)                     # how good the model is at ranking FAILED above PASSED (good metric for imbalanced datasets)

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

# Clustering finds structure WITHOUT using the inspection result(PASSED/FAILED).
# This helps discover hidden risk profiles based only on features.

X_processed = preprocessor.fit_transform(X)

# Initialize K-Means with 4 clusters
# n_init=10 runs the algorithm multiple times to avoid poor random initialization
# random_state ensures reproducible clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# Assign each observation to a cluster (0, 1, 2, or 3)
df["cluster"] = kmeans.fit_predict(X_processed)

# Analyze clusters by checking the average failure rate in each one
# This helps interpret which clusters represent higher-risk profiles
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

# contamination=0.02 assumes that about 2% of observations are anomalies
iso = IsolationForest(contamination=0.02, random_state=42)
# Predict anomalies: 1 = normal observation, -1 = anomaly
df["anomaly"] = iso.fit_predict(X_processed)

# Show how many anomalies were detected
print("Anomaly distribution (-1 = anomaly):")
print(df["anomaly"].value_counts(), "\n")

# Display a few anomalous observations to inspect their characteristics
print("Sample anomalies:")
print(df[df["anomaly"] == -1][
    ["Public Vehicle Type", "Vehicle Make", "vehicle_age", "Result"]
].head())

# ============================================================
# 11. ABLATION STUDY – TIME FEATURES (COMPARISON ONLY)
# ============================================================

print("\nStep 11: Ablation study – including time features (comparison only)...\n")

# My hypothesis:
# “Maybe inspections scheduled at certain times are more likely to fail.” 
# Examples we might expect:
 #inspections early in the morning / late in the day
 #inspections at the end of the month
 #inspections at the end of the year
#This section checks if that intuition is true or false.

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

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n================ PROJECT COMPLETED SUCCESSFULLY ================\n")
