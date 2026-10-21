import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_url_model.joblib"
SAMPLES_PATH = BASE_DIR / "demo_samples.csv"

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def normalize_url(url):
    url = url.strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    return url

st.title("🔐 Phishing Website Detector")

st.write(
    "Analyze a website URL using a machine learning model "
    "trained on the UCI PhiUSIIL phishing URL dataset."
)

st.info(
    "This is an educational ML prototype. Its predictions "
    "are estimates, not proof that a website is safe."
)

# Load correctly classified held-out examples for the expo
if SAMPLES_PATH.exists():
    samples = pd.read_csv(SAMPLES_PATH).dropna(
        subset=["url", "actual"]
    )

    options = ["Enter a URL yourself"]

    for i, row in samples.reset_index(drop=True).iterrows():
        options.append(
            f"Demo {i + 1}: {row['actual']}"
        )

    choice = st.selectbox(
        "Try a demonstration example (optional)",
        options
    )

    default_url = ""

    if choice != "Enter a URL yourself":
        index = options.index(choice) - 1
        default_url = str(
            samples.reset_index(drop=True).iloc[index]["url"]
        )
else:
    default_url = ""

url_input = st.text_input(
    "Enter the website URL",
    value=default_url,
    placeholder="https://example.com"
)

if st.button("Check Website", type="primary", use_container_width=True):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        url = normalize_url(url_input)

        try:
            prediction = model.predict([url])[0]
            probabilities = model.predict_proba([url])[0]
            classes = list(model.classes_)

            score = float(
                probabilities[classes.index(prediction)]
            )

            st.divider()

            if prediction == "Phishing":
                st.error("⚠️ Potentially Phishing")
            else:
                st.success("✅ Likely Legitimate")

            st.metric(
                "Model score for predicted class",
                f"{score * 100:.1f}%"
            )

            st.caption(
                "This score is not a guarantee of safety. "
                "A new or unfamiliar URL may be misclassified."
            )

            with st.expander("What did the model analyze?"):
                st.write("Submitted URL:", url)
                st.write(
                    "The model analyzes character patterns "
                    "learned from labeled URLs. It does not "
                    "visit the website or inspect its live content."
                )

        except Exception as error:
            st.error("Prediction failed. Please check the input.")
            st.exception(error)
