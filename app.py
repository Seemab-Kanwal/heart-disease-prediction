import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the 4 details below to predict the risk of heart disease.")

model = joblib.load("heart_model.pkl")
le_sex = joblib.load("sex_encoder.pkl")

age = st.slider("Age (years)", 20, 90, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4],
    format_func=lambda x: {1: "1 - Typical angina", 2: "2 - Atypical angina",
                           3: "3 - Non-anginal pain", 4: "4 - Asymptomatic"}[x],
)
thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)

if st.button("Predict"):
    sex_num = le_sex.transform([sex])[0]
    row = pd.DataFrame([[age, sex_num, cp, thalach]],
                       columns=["age", "sex", "cp", "thalach"])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"⚠️ Heart Disease likely (probability {prob*100:.1f}%)")
    else:
        st.success(f"✅ No Heart Disease likely (probability of disease {prob*100:.1f}%)")
    st.caption("Educational project only - not a medical diagnosis.")
