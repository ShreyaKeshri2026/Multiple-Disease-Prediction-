import streamlit as st
import pickle
import numpy as np

# Load models
@st.cache_resource
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load all models
heart_model = load_model("model2.pkl")
diabetes_model = load_model("model1.pkl")
cancer_model = load_model("model.pkl")

# Streamlit UI
st.title("Multiple Disease Prediction System 🏥")

# Disease selection
disease = st.selectbox("Select a disease to predict:", ["Heart Disease", "Diabetes", "Breast Cancer"])

# Define input fields based on disease type
if disease == "Heart Disease":
    st.header("Heart Disease Prediction 💓")
    feature_names = [ "Age", "Sex (1=male, 0=female)", "Chest Pain Type (1-4)", 
    "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar (1=Yes, 0=No)",
    "Resting Electrocardiographic Results (0-2)", "Max Heart Rate Achieved",
    "Exercise Induced Angina (1=Yes, 0=No)", 
    "ST Depression Induced by Exercise Relative to Rest",
    "Slope of Peak Exercise ST Segment", 
    "Number of Major Vessels Colored by Fluoroscopy (0-3)", 
    "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect"]

    # User input for heart disease
    features = [st.number_input(f"{name}", value=0.0) for name in feature_names]

    if st.button("Predict Heart Disease"):
        input_data = np.array([features]).reshape(1, -1)
        prediction = heart_model.predict(input_data)
        st.success("No Heart Disease detected! ✅" if prediction[0] == 0 else "Heart Disease detected! ⚠️")

elif disease == "Diabetes":
    st.header("Diabetes Prediction 🍬")
    feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                     "Insulin", "BMI", "Diabetes Pedigree", "Age"]

    # User input for diabetes
    features = [st.number_input(f"{name}", value=0.0) for name in feature_names]

    if st.button("Predict Diabetes"):
        input_data = np.array([features]).reshape(1, -1)
        prediction = diabetes_model.predict(input_data)
        st.success("No Diabetes detected! ✅" if prediction[0] == 0 else "Diabetes detected! ⚠️")

elif disease == "Breast Cancer":
    st.header("Breast Cancer Prediction 🎗️")
    feature_names = ["Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
                 "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
                 "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
                 "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
                 "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
                 "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"]

    # User input for breast cancer
    features = [st.number_input(f"{name}", value=0.0) for name in feature_names]

    if st.button("Predict Breast Cancer"):
        input_data = np.array([features]).reshape(1, -1)
        prediction = cancer_model.predict(input_data)
        st.success("No Breast Cancer detected! ✅" if prediction[0] == 0 else "Breast Cancer detected! ⚠️")

# Run the app with: streamlit run app.py
