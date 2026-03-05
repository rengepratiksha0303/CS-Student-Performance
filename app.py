import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("🎓 Student Career Prediction")

# Load models safely
knn = None
ann = None
cnn = None
scaler = None

try:
    knn = pickle.load(open("knn_model.pkl","rb"))
except:
    st.warning("knn_model.pkl not found")

try:
    ann = load_model("ann_model.h5")
except:
    st.warning("ann_model.h5 not found")

try:
    cnn = load_model("cnn_model.h5")
except:
    st.warning("cnn_model.h5 not found")

try:
    scaler = pickle.load(open("scaler.pkl","rb"))
except:
    st.warning("scaler.pkl not found")


# Input fields
age = st.number_input("Age",18,30)

gender = st.selectbox("Gender",["Male","Female"])

gpa = st.number_input("GPA",0.0,10.0)

programming = st.slider("Programming Skill",1,10)

algorithms = st.slider("Algorithms Skill",1,10)

math = st.slider("Math Skill",1,10)

communication = st.slider("Communication Skill",1,10)

teamwork = st.slider("Teamwork",1,10)

leadership = st.slider("Leadership",1,10)

gender = 1 if gender=="Male" else 0

features = np.array([[age,gender,gpa,programming,
                      algorithms,math,
                      communication,teamwork,
                      leadership]])

if scaler:
    features = scaler.transform(features)

model_choice = st.selectbox("Model",["KNN","ANN","CNN"])

if st.button("Predict"):

    prediction = None

    if model_choice=="KNN" and knn:
        prediction = knn.predict(features)

    elif model_choice=="ANN" and ann:
        prediction = np.argmax(ann.predict(features),axis=1)

    elif model_choice=="CNN" and cnn:
        features = features.reshape(features.shape[0],features.shape[1],1)
        prediction = np.argmax(cnn.predict(features),axis=1)

    if prediction is not None:
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.error("Model file missing")
