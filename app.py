import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🎓",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
REGRESSION_MODEL = ROOT / "Mental_Health_Regression_Model.pkl"
CLASSIFICATION_MODEL = ROOT / "Mental_Health_Classification_Model.pkl"
METADATA_FILE = ROOT / "model_metadata.json"

st.title("🎓 Student Social Media & Mental Health Prediction")
st.write(
    "Enter student lifestyle and social-media usage details to estimate a "
    "mental-health score and its project-defined score category."
)


@st.cache_resource
def load_assets():
    missing = [
        p.name
        for p in (REGRESSION_MODEL, CLASSIFICATION_MODEL, METADATA_FILE)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing)
            + ". First run train_models.py with the dataset."
        )
    reg = joblib.load(REGRESSION_MODEL)
    clf = joblib.load(CLASSIFICATION_MODEL)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return reg, clf, metadata


try:
    regression_model, classification_model, meta = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Keep the dataset in this folder and run `python train_models.py`. "
        "That creates both model files and model_metadata.json."
    )
    st.stop()

with st.sidebar:
    st.header("About this project")
    st.write("Target: `Mental_Health_Score`")
    st.write("Models: tuned Random Forest Regressor and tuned Random Forest Classifier.")
    st.caption(
        "Academic demonstration only. The predicted category is derived from "
        "dataset score quantiles; it is not a clinical diagnosis."
    )

st.subheader("Enter student details")
with st.form("student_input_form"):
    left, middle, right = st.columns(3)

    numeric = meta["numeric_defaults"]
    ranges = meta["numeric_ranges"]
    categorical = meta["categorical_options"]

    with left:
        age = st.number_input(
            "Age",
            min_value=int(ranges["Age"][0]),
            max_value=int(ranges["Age"][1]),
            value=int(numeric["Age"]),
            step=1,
        )
        avg_usage = st.number_input(
            "Average daily social media usage (hours)",
            min_value=0.0,
            max_value=max(24.0, float(ranges["Avg_Daily_Usage_Hours"][1])),
            value=float(numeric["Avg_Daily_Usage_Hours"]),
            step=0.5,
        )
        unlocks = st.number_input(
            "Daily phone unlocks",
            min_value=0,
            max_value=max(500, int(ranges["Daily_Unlocks"][1])),
            value=int(numeric["Daily_Unlocks"]),
            step=1,
        )
        study = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=float(numeric["Study_Hours"]),
            step=0.5,
        )

    with middle:
        activity = st.number_input(
            "Physical activity hours per day",
            min_value=0.0,
            max_value=24.0,
            value=float(numeric["Physical_Activity_Hours"]),
            step=0.5,
        )
        sleep = st.number_input(
            "Sleep hours per night",
            min_value=0.0,
            max_value=24.0,
            value=float(numeric["Sleep_Hours_Per_Night"]),
            step=0.5,
        )
        gender = st.selectbox("Gender", categorical["Gender"])
        academic = st.selectbox("Academic level", categorical["Academic_Level"])
        stress = st.selectbox("Stress level", ["Low", "Medium", "High", "Very High"])

    with right:
        platform = st.selectbox("Most-used platform", categorical["Most_Used_Platform"])
        purpose = st.selectbox("Purpose of use", categorical["Purpose_Of_Use"])
        # Deduplicate while preserving order so "Other" never appears twice
        country_options = list(dict.fromkeys(meta["top_countries"] + ["Other"]))
        country = st.selectbox("Country", country_options)
        st.caption("Countries outside the dataset's 10 most frequent are grouped as Other.")

    submitted = st.form_submit_button("Predict mental health score", type="primary")

if submitted:
    input_row = pd.DataFrame(
        [
            {
                "Study_Hours": study,
                "Age": age,
                "Avg_Daily_Usage_Hours": avg_usage,
                "Daily_Unlocks": unlocks,
                "Physical_Activity_Hours": activity,
                "Sleep_Hours_Per_Night": sleep,
                "Stress_Level": stress,
                "Gender": gender,
                "Academic_Level": academic,
                "Most_Used_Platform": platform,
                "Purpose_Of_Use": purpose,
                "Grouped_country": country,
            }
        ]
    )

    # Keep the exact feature order used to train the pipelines.
    input_row = input_row[meta["feature_columns"]]
    score = float(regression_model.predict(input_row)[0])
    category = str(classification_model.predict(input_row)[0])

    st.divider()
    st.subheader("Prediction result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Mental Health Score", f"{score:.2f}")
    c2.metric("Predicted Score Category", category)

    st.caption(
        "The Low / Moderate / High label is a machine-learning category created "
        "from score quantiles in the training dataset. It does not indicate or "
        "diagnose a mental-health condition."
    )
