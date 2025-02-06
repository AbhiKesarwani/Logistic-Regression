import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("Heart_disease_model.joblib")  # Update with your model file

model = load_model()

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🩺 Prediction"])

### 1️⃣ HOME PAGE ###
if page == "🏠 Home":
    st.title("❤️ Heart Disease Prediction App")
    st.write("This app predicts the likelihood of heart disease based on patient details.")

    # Dataset Insights
    dataset = load_iris()  # Replace with heart disease dataset
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    st.subheader("📊 Dataset Insights")
    st.write(df.describe())

    # Example Graph
    fig, ax = plt.subplots()
    ax.hist(df["sepal length (cm)"], bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Example: Feature Distribution (Replace with Heart Data)")
    st.pyplot(fig)

    st.warning("⚠️ Disclaimer: This is a project and should not be used for real medical diagnosis.")

### 2️⃣ PREDICTION PAGE ###
elif page == "🩺 Prediction":
    st.title("🩺 Heart Disease Prediction")

    # Sidebar Input Form
    st.sidebar.header("Patient Details")
    age = st.sidebar.slider("Age", 1, 120, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])

    # Convert sex to binary
    sex = 1 if sex == "Male" else 0

    # DataFrame for input
    input_data = pd.DataFrame({
        "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
        "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
        "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
    })

    # Show input data
    st.subheader("📋 Entered Patient Data")
    st.write(input_data)

    # Predict button
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("📊 Prediction Result")
        if prediction[0] == 1:
            st.error("❌ High risk of heart disease detected!")
        else:
            st.success("✅ Low risk of heart disease.")

        # Probability Chart
        st.subheader("📈 Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["No Heart Disease", "Heart Disease"], prediction_proba[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # Risk Level Alerts
        if prediction_proba[0][1] > 0.7:
            st.warning("⚠️ High risk! Consult a doctor immediately.")
        elif prediction_proba[0][1] > 0.4:
            st.info("ℹ️ Moderate risk. Consider a check-up.")
        else:
            st.success("🌟 Low risk. Maintain a healthy lifestyle!")
