"""Train deployment models using the project's dataset (.csv, .xlsx, or .xls).

Run:
    python train_models.py

Input:
    Student Social Media And Mental Health Impact.csv (or .xlsx / .xls)

Outputs:
    Mental_Health_Regression_Model.pkl
    Mental_Health_Classification_Model.pkl
    model_metadata.json
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent

CANDIDATE_FILES = [
    ROOT / "Student Social Media And Mental Health Impact.csv",
    ROOT / "Student Social Media And Mental Health Impact.xlsx",
    ROOT / "Student Social Media And Mental Health Impact.xls",
]

DATA_FILE = next((p for p in CANDIDATE_FILES if p.exists()), None)
if DATA_FILE is None:
    raise FileNotFoundError(
        "Dataset not found. Place 'Student Social Media And Mental Health Impact.csv' "
        "(or .xlsx / .xls) in the same folder as train_models.py."
    )

if DATA_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.read_excel(DATA_FILE)

df.columns = df.columns.astype(str).str.strip()

numeric_columns = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
]
for col in numeric_columns:
    if col not in df.columns:
        raise KeyError(f"Required column missing from dataset: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")

required_columns = [
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Stress_Level",
]
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column missing from dataset: {col}")

df = df.drop_duplicates().copy()
df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
df["Study_Hours"] = df["Study_Hours"].clip(lower=0)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Mental_Health_Score"]).copy()

top_countries = df["Country"].dropna().value_counts().head(10).index.tolist()
df["Grouped_country"] = df["Country"].apply(
    lambda value: value if pd.notna(value) and value in top_countries else "Other"
)

study_cols = ["Study_Hours"]
other_numeric_cols = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]
ordinal_cols = ["Stress_Level"]
nominal_cols = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]
feature_columns = study_cols + other_numeric_cols + ordinal_cols + nominal_cols

study_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ]
)
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
ordinal_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OrdinalEncoder(
                categories=[["Low", "Medium", "High", "Very High"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)
nominal_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("Study_Hours", study_pipeline, study_cols),
        ("Numeric", numeric_pipeline, other_numeric_cols),
        ("Stress", ordinal_pipeline, ordinal_cols),
        ("Categorical", nominal_pipeline, nominal_cols),
    ]
)

X = df[feature_columns].copy()
y = df["Mental_Health_Score"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)

regression_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ]
)
regression_params = {
    "random_forest__n_estimators": [100, 200],
    "random_forest__max_depth": [5, 10, None],
    "random_forest__min_samples_split": [2, 5],
    "random_forest__min_samples_leaf": [1, 2],
}
regression_search = RandomizedSearchCV(
    regression_pipeline,
    param_distributions=regression_params,
    n_iter=10,
    cv=3,
    scoring="r2",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
regression_search.fit(X_train, y_train)
joblib.dump(regression_search.best_estimator_, ROOT / "Mental_Health_Regression_Model.pkl", compress=3)
print("Saved tuned regression model.")

if df["Mental_Health_Score"].nunique() < 3:
    raise ValueError("At least three distinct score values are needed for classification.")
y_class = pd.qcut(
    df["Mental_Health_Score"].rank(method="first"),
    q=3,
    labels=["Low", "Moderate", "High"],
)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.30, random_state=RANDOM_STATE, stratify=y_class
)

classification_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
    ]
)
classification_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 8, 12],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}
classification_search = GridSearchCV(
    classification_pipeline,
    param_grid=classification_params,
    scoring="f1_weighted",
    cv=3,
    n_jobs=-1,
)
classification_search.fit(X_train_c, y_train_c)
joblib.dump(
    classification_search.best_estimator_,
    ROOT / "Mental_Health_Classification_Model.pkl",
    compress=3,
)
print("Saved tuned classification model.")


def safe_number(value, fallback):
    return fallback if pd.isna(value) else float(value)


numeric_defaults = {}
numeric_ranges = {}
for col in [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]:
    numeric_defaults[col] = safe_number(df[col].median(), 0)
    lo, hi = df[col].min(), df[col].max()
    if pd.isna(lo) or pd.isna(hi):
        lo, hi = 0, 24
    if lo == hi:
        hi = lo + 1
    numeric_ranges[col] = [float(lo), float(hi)]

categorical_options = {}
for col in ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use"]:
    vals = sorted(df[col].dropna().astype(str).unique().tolist())
    categorical_options[col] = vals or ["Unknown"]

metadata = {
    "feature_columns": feature_columns,
    "top_countries": [str(x) for x in top_countries],
    "numeric_defaults": numeric_defaults,
    "numeric_ranges": numeric_ranges,
    "categorical_options": categorical_options,
}
(ROOT / "model_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print("Saved model_metadata.json")
print("\nTraining completed successfully.")
