import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
import os

# App title
st.title('Diabetes Prediction and Analysis - Pima Indians Dataset')

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    
    # Replace zeros with median per class
    cols_with_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'Glucose', 'BMI']
    for col in cols_with_zeros:
        df[col] = df.groupby('Outcome')[col].transform(
            lambda x: x.mask(x == 0, x[x != 0].median())
        )
    return df

df = load_data()

# Load the pre-trained model safely
model_path = 'final_model_acc86.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please train the model first.")
    st.stop()

try:
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error("Failed to load the model.")
    st.exception(e)
    st.stop()

# Prepare the scaler based on training data
scaler = StandardScaler()
x = df.drop('Outcome', axis=1)
scaler.fit(x)

# Prediction form
st.subheader("Make a Prediction")
with st.form("prediction_form"):
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
    bp = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=846, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=21, max_value=100, value=30)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    if prob > 0.5:
        st.error(f'Prediction: Positive for Diabetes (Probability: {prob:.2f})')
    else:
        st.success(f'Prediction: Negative for Diabetes (Probability: {1 - prob:.2f})')
