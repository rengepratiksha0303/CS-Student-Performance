import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

st.title("🎓 Student Career Prediction App")

st.write("Predict the future career of a CS student using ML & DL models")

# Load models
knn = pickle.load(open("knn_model.pkl", "rb"))
ann = load_model("ann_model.h5")
cnn = load_model("cnn_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# User Inputs
age = st.number_input("Age", 18, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

gpa = st.number_input("GPA", 0.0, 10.0)

programming = st.slider("Programming Skill (1-10)",1,10)

algorithms = st.slider("Algorithms Skill (1-10)",1,10)

math = st.slider("Mathematics Skill (1-10)",1,10)

communication = st.slider("Communication Skill (1-10)",1,10)

teamwork = st.slider("Teamwork Skill (1-10)",1,10)

leadership = st.slider("Leadership Skill (1-10)",1,10)

# Encode gender
gender = 1 if gender == "Male" else 0

# Feature array
features = np.array([[age, gender, gpa, programming,
                      algorithms, math,
                      communication, teamwork,
                      leadership]])

# Scale
features = scaler.transform(features)

# Model Selection
model_choice = st.selectbox(
    "Select Model",
    ["KNN", "ANN", "CNN"]
)

if st.button("Predict Career"):

    if model_choice == "KNN":
        prediction = knn.predict(features)

    elif model_choice == "ANN":
        prediction = np.argmax(ann.predict(features), axis=1)

    else:
        features_cnn = features.reshape(features.shape[0], features.shape[1], 1)
        prediction = np.argmax(cnn.predict(features_cnn), axis=1)

    st.success(f"Predicted Career Category: {prediction[0]}")
