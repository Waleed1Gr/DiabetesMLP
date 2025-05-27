# DiabetesMLP

A machine learning project using a Multi-Layer Perceptron (MLP) model to predict diabetes based on medical diagnostic data. This project also includes a Streamlit web app for user-friendly predictions.

## 🔬 Project Overview

This project focuses on predicting whether a patient has diabetes using the **Pima Indians Diabetes Dataset**. The model is built using Keras with TensorFlow backend and saved as an `.h5` file for deployment. The frontend is developed using Streamlit to provide interactive model predictions.

## 📁 Repository Structure
<pre>
DiabetesMLP/
├── DiabetesMLP.ipynb # Jupyter Notebook for data processing and training
├── Model.h5 # Trained MLP model
├── Streamlitt_app/
│ ├── app.py # Streamlit web app
│ └── requirements.txt # List of dependencies for the app
└── README.md # Project documentation
</pre>

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: 8 medical predictor variables
- Target: Binary outcome (1 = diabetic, 0 = non-diabetic)

## 🤖 Model Details

- Type: Multi-Layer Perceptron (MLP)
- Framework: TensorFlow / Keras
- Architecture: 3 hidden layers with ReLU activation
- Output: Sigmoid activation for binary classification

## 🚀 How to Run the App

### 1. Clone the repository
```
git clone https://github.com/Waleed1Gr/DiabetesMLP.git
cd DiabetesMLP/Streamlitt_app
```
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the Streamlit app
streamlit run app.py
✅ Requirements
Python 3.7+

TensorFlow

Keras

Pandas

Scikit-learn

Streamlit

NumPy

Streamlit

You can install all requirements using the provided requirements.txt.

### 📈 Results
The model achieves good accuracy on test data, and the Streamlit app allows for real-time predictions by inputting patient data through sliders and forms.

### 👨‍💻 Team Members
This project was developed by a team of five students:

وليد القرافي

ماجد السرواني

نوره الوابل

عبدالرحمن ال عباس

رغد القحطاني
