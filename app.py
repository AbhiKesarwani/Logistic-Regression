import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("Heart_disease_model.joblib")  # Replace with your model file path
    return model

model = load_model()

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered",
)

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1579546929662-711aa81148cf?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #ff1c1c;
    }
    .stMarkdown h1 {
        color: #ff4b4b;
    }
    .stMarkdown h2 {
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the app
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Input fields for user data
st.sidebar.header("Patient Details")
age = st.sidebar.slider("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", options=[0, 1])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3])

# Convert sex to binary
sex = 1 if sex == "Male" else 0

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Display the input data
st.subheader("📋 Patient Input Data")
st.write(input_data)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.error("❌ The model predicts that the patient has heart disease.")
    else:
        st.success("✅ The model predicts that the patient does not have heart disease.")

    # Visualization: Prediction Probabilities
    st.subheader("📈 Prediction Probability")
    fig, ax = plt.subplots()
    ax.bar(["No Heart Disease", "Heart Disease"], prediction_proba[0], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Additional Insights
    st.subheader("💡 Insights")
    st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")

    # Emoji-based feedback
    if prediction_proba[0][1] > 0.7:
        st.warning("⚠️ High risk of heart disease detected. Please consult a doctor.")
    elif prediction_proba[0][1] > 0.4:
        st.info("ℹ️ Moderate risk of heart disease detected. Consider a check-up.")
    else:
        st.success("🌟 Low risk of heart disease. Keep up the good health habits!")
