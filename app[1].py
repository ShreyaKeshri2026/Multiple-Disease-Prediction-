import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_pickle("breast_cancer_data.pkl")

# Title
st.title("Breast Cancer Data Explorer")

# Load data
loaded_data = pd.read_pickle("breast_cancer_data.pkl")

# Display dataset
st.write("### Breast Cancer Dataset Preview")
st.dataframe(loaded_data)

# Display dataset statistics
st.write("### Dataset Statistics")
st.write(loaded_data.describe())

# Option to show dataset shape
if st.checkbox("Show dataset shape"):
    st.write(f"Shape of dataset: {loaded_data.shape}")
    
    
# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("Breast Cancer Prediction")

# User Input Fields
st.write("Enter feature values:")

input_features = []
feature_names = ["Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
                 "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
                 "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
                 "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
                 "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
                 "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"]

for feature in feature_names:
    value = st.number_input(feature, min_value=0.0, max_value=1000.0, step=0.01)
    input_features.append(value)

# Convert input to NumPy array and reshape
input_data = np.array(input_features).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.write("The Breast Cancer is **Malignant**")
    else:
        st.write("The Breast Cancer is **Benign**")