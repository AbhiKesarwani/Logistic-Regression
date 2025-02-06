import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

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
    st.write("""
    This project predicts the likelihood of heart disease based on patient details. 
    It uses Machine Learning to analyze physical and medical test data. 

    🚨 **Why is this important?**  
    Diagnosing heart disease often requires **invasive and expensive** tests. 
    This model aims to help identify **high-risk** patients early, reducing unnecessary procedures.
    """)

    # Dataset Information
    st.header("📊 Dataset Overview")
    st.write("""
    The dataset contains **14 medical attributes** collected from patients. 
    It includes blood sample results and exercise test results. 

    - **Age**: Patient's age in years  
    - **Sex**: 0 = Female, 1 = Male  
    - **Chest Pain Type (cp)**:  
      - 0 = Typical Angina  
      - 1 = Atypical Angina  
      - 2 = Non-anginal Pain  
      - 3 = Asymptomatic  
    - **Resting Blood Pressure (trestbps)**: mm Hg  
    - **Serum Cholesterol (chol)**: mg/dl  
    - **Fasting Blood Sugar (fbs)**: 1 = >120 mg/dl, 0 = Normal  
    - **Resting ECG Results (restecg)**:  
      - 0 = Normal  
      - 1 = ST-T Wave Abnormality  
      - 2 = Left Ventricular Hypertrophy  
    - **Maximum Heart Rate (thalach)**  
    - **Exercise Induced Angina (exang)**: 1 = Yes, 0 = No  
    - **ST Depression (oldpeak)**  
    - **Slope of ST Segment (slope)**: 0 = Upsloping, 1 = Flat, 2 = Downsloping  
    - **Number of Major Vessels (ca)**: 0 to 3, detected by fluoroscopy  
    - **Thalassemia (thal)**:  
      - 3 = Normal  
      - 6 = Fixed Defect  
      - 7 = Reversible Defect  
    - **Target (heart disease presence)**:  
      - 0 = No heart disease  
      - 1 = Heart disease detected  
    """)

    # Load example dataset (replace with real heart dataset)
    dataset = load_iris()  # Replace this with actual heart disease dataset
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    st.subheader("📌 Dataset Statistics")
    st.write(df.describe())

    # Example Graph
    st.subheader("📈 Feature Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["sepal length (cm)"], bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Example: Feature Distribution (Replace with Heart Data)")
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Model Performance - Confusion Matrix")
    
    # Generate a dummy confusion matrix (Replace this with actual test data)
    y_true = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  # Replace with real labels
    y_pred = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]  # Replace with model predictions

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Heart Disease"], yticklabels=["No Disease", "Heart Disease"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.warning("⚠️ **Disclaimer**: This is a project and should not be used for real medical diagnosis.")

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
    thal = st.sidebar.selectbox("Thalassemia", [3, 6, 7])

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
