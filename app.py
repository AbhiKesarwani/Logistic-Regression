import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Set page config FIRST!
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

# ✅ Load the trained model safely
@st.cache_resource
def load_model():
    try:
        return joblib.load("Heart_disease_model.joblib")
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

model = load_model()

# ✅ Custom CSS for modern styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #ffe6e6, #ffcccc);
        padding: 20px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1c1c;
    }
    h1 {
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ✅ Title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# ✅ Sidebar Inputs
st.sidebar.header("🩺 Patient Details")
age = st.sidebar.slider("Age", 1, 120, 50)
sex = 1 if st.sidebar.radio("Sex", ["Male", "Female"]) == "Male" else 0
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])

# ✅ DataFrame for model input
input_data = pd.DataFrame({
    "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
    "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
    "exang": [exang], "oldpeak": [oldpeak], "slope": [slope],
    "ca": [ca], "thal": [thal]
})

# ✅ Display input data
st.subheader("📋 Patient Input Data")
st.write(input_data)

# ✅ Prediction Button
if st.button("🔍 Predict") and model is not None:
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    # ✅ Display Prediction Result
    st.subheader("📊 Prediction Result")
    result_text = "❌ High chance of heart disease detected!" if prediction[0] == 1 else "✅ No heart disease detected."
    st.markdown(f"### {result_text}")

    # ✅ Probability Visualization
    fig, ax = plt.subplots()
    labels = ["No Heart Disease", "Heart Disease"]
    colors = ["green", "red"]
    ax.pie(prediction_proba, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Prediction Probability")
    st.pyplot(fig)

    # ✅ Risk Assessment
    st.subheader("💡 Risk Assessment")
    st.write(f"🔴 Probability of having heart disease: **{prediction_proba[1]:.2f}**")
    st.write(f"🟢 Probability of NOT having heart disease: **{prediction_proba[0]:.2f}**")

    # ✅ Risk Category Alerts
    if prediction_proba[1] > 0.7:
        st.warning("⚠️ High risk detected! Consult a doctor immediately.")
    elif prediction_proba[1] > 0.4:
        st.info("ℹ️ Moderate risk. Consider a medical check-up.")
    else:
        st.success("🌟 Low risk! Keep up a healthy lifestyle.")
