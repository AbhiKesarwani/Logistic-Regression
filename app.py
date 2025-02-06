import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("Heart_disease_model.joblib")  # Update with your model file
    return model

model = load_model()

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Prediction"])

if page == "🏠 Home":
    # Title and Introduction
    st.title("❤️ Heart Disease Prediction App")
    st.write("This application predicts the likelihood of heart disease based on patient data using a trained machine learning model.")

    # Dataset Overview
    st.header("📊 Dataset Insights")
    df = pd.read_csv("heart.csv")  # Update with actual dataset path
    st.write("Sample of the dataset:")
    st.dataframe(df.head())

    # Data Distribution
    st.subheader("Feature Distributions")
    fig = px.histogram(df, x="age", title="Age Distribution", nbins=20, color_discrete_sequence=["#FF4B4B"])
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Feature Correlation")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Disclaimer
    st.warning("⚠️ This app is for educational purposes only and should not be used for medical diagnosis.")

elif page == "🔍 Prediction":
    st.title("🔍 Heart Disease Prediction")
    st.write("Enter the patient's details to predict the likelihood of heart disease.")

    # User Input
    st.sidebar.header("Patient Details")
    age = st.sidebar.slider("Age", 1, 120, 50)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.radio("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])

    # Convert sex to binary
    sex = 1 if sex == "Male" else 0

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps], "chol": [chol],
        "fbs": [fbs], "restecg": [restecg], "thalach": [thalach], "exang": [exang],
        "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
    })

    # Show Input Data
    st.subheader("📋 Patient Input Data")
    st.write(input_data)

    # Predict Button
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("📊 Prediction Result")
        if prediction[0] == 1:
            st.error("❌ High risk of heart disease detected.")
        else:
            st.success("✅ No signs of heart disease detected.")

        # Visualization: Prediction Probability
        st.subheader("📈 Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["No Heart Disease", "Heart Disease"], prediction_proba[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        if hasattr(model, 'coef_'):
            feature_importance = abs(model.coef_[0])
            feature_df = pd.DataFrame({"Feature": input_data.columns, "Importance": feature_importance})
            feature_df = feature_df.sort_values(by="Importance", ascending=False)
            fig = px.bar(feature_df, x="Importance", y="Feature", orientation="h", title="Feature Importance", color="Importance")
            st.plotly_chart(fig)

        # Insights
        st.subheader("💡 Insights")
        st.write(f"Probability of heart disease: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of no heart disease: {prediction_proba[0][0]:.2f}")

        # Risk Level Indicator
        if prediction_proba[0][1] > 0.7:
            st.warning("⚠️ High risk detected. Consult a doctor immediately.")
        elif prediction_proba[0][1] > 0.4:
            st.info("ℹ️ Moderate risk detected. Consider a health check-up.")
        else:
            st.success("🌟 Low risk detected. Maintain a healthy lifestyle!")
