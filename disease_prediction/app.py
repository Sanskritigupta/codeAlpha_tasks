import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.write("A machine-learning demonstration using selected health-related input features.")
st.warning("Educational demo using synthetic data only. It cannot diagnose disease or replace a qualified medical professional.")

model_path = Path(__file__).with_name("heart_model.pkl")
if not model_path.exists():
    st.info("Model file not found. Run `python train_model.py` first.")
    st.stop()
model = joblib.load(model_path)

with st.form("health_form"):
    age = st.slider("Age", 18, 100, 45)
    sex = st.selectbox("Sex (dataset encoding)", ["Female (0)", "Male (1)"])
    cp = st.selectbox("Chest pain type (encoded 0–3)", [0, 1, 2, 3])
    bp = st.slider("Resting blood pressure (mm Hg)", 80, 220, 120)
    chol = st.slider("Cholesterol (mg/dL)", 100, 600, 220)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dL?", ["No (0)", "Yes (1)"])
    hr = st.slider("Maximum heart rate", 60, 220, 150)
    exang = st.selectbox("Exercise-induced angina?", ["No (0)", "Yes (1)"])
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    submitted = st.form_submit_button("Predict")
if submitted:
    row = pd.DataFrame([{
        "age": age, "sex": 0 if sex.startswith("Female") else 1,
        "chest_pain_type": cp, "resting_bp": bp, "cholesterol": chol,
        "fasting_blood_sugar": 0 if fbs.startswith("No") else 1,
        "max_heart_rate": hr, "exercise_angina": 0 if exang.startswith("No") else 1,
        "oldpeak": oldpeak
    }])
    pred = int(model.predict(row)[0])
    if pred == 1:
        st.error("Model output: Risk pattern detected in this demo.")
    else:
        st.success("Model output: No risk pattern detected in this demo.")
    st.caption("This is not a medical diagnosis. Consult a healthcare professional for medical concerns.")
