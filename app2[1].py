import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("model2.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.title("Heart Disease Prediction App 💓")

st.write("Enter the following details to check for heart disease risk.")

# Creating input fields for the features in the dataset
feature_names = [
    "Age", "Sex (1=male, 0=female)", "Chest Pain Type (1-4)", 
    "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar (1=Yes, 0=No)",
    "Resting Electrocardiographic Results (0-2)", "Max Heart Rate Achieved",
    "Exercise Induced Angina (1=Yes, 0=No)", 
    "ST Depression Induced by Exercise Relative to Rest",
    "Slope of Peak Exercise ST Segment", 
    "Number of Major Vessels Colored by Fluoroscopy (0-3)", 
    "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"
    
]

features = []
for name in feature_names:
    features.append(st.number_input(f"{name}", value=0.0))

# Predict button
if st.button("Predict"):
    input_data = np.array([features]).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 The person is at **Risk of Heart Disease**.")
    else:
        st.success("🟢 The person is **Not at Risk of Heart Disease**.")
