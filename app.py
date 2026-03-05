import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

st.title("🎓 Student Career Prediction App")

# Load Models Safely
try:
    knn = pickle.load(open("knn_model.pkl", "rb"))
except:
    st.error("knn_model.pkl file not found")
    knn = None

try:
    ann = load_model("ann_model.h5")
except:
    st.error("ann_model.h5 file not found")
    ann = None

try:
    cnn = load_model("cnn_model.h5")
except:
    st.error("cnn_model.h5 file not found")
    cnn = None

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    st.error("scaler.pkl file not found")
    scaler = None


# Input Fields
age = st.number_input("Age", 18, 30)

gender = st.selectbox("Gender", ["Male","Female"])

gpa = st.number_input("GPA", 0.0, 10.0)

programming = st.slider("Programming Skill",1,10)

algorithms = st.slider("Algorithms Skill",1,10)

math = st.slider("Math Skill",1,10)

communication = st.slider("Communication Skill",1,10)

teamwork = st.slider("Teamwork",1,10)

leadership = st.slider("Leadership",1,10)


# Encode gender
gender = 1 if gender == "Male" else 0


features = np.array([[age,gender,gpa,programming,algorithms,
                      math,communication,teamwork,leadership]])

if scaler:
    features = scaler.transform(features)


model_choice = st.selectbox("Select Model",["KNN","ANN","CNN"])


if st.button("Predict"):

    if model_choice=="KNN" and knn:
        prediction = knn.predict(features)

    elif model_choice=="ANN" and ann:
        prediction = np.argmax(ann.predict(features),axis=1)

    elif model_choice=="CNN" and cnn:
        features = features.reshape(features.shape[0],features.shape[1],1)
        prediction = np.argmax(cnn.predict(features),axis=1)

    else:
        st.error("Model file missing")

    st.success(f"Prediction: {prediction}")
