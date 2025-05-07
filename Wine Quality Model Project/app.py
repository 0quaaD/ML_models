import streamlit as st
import pandas as pd
import numpy as np
import pickle


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open('./model/wine_model.pkl','rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Wine Quality Classifier", layout='centered')
st.title("Wine Quality Classifier")
st.markdown("Predict whether a red wine is **Good (1)** or **Bad (0)** based on its chemical properties.")

st.header("Input Wine Features")

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

input_data = []
for feat in feature_names:
    val = st.slider(f"{feat}", min_value = 0.0, max_value = 15.0, step = 0.05, value = 5.0)
    input_data.append(val)


# if you want to test the confidence of the model based on the real data then you may use this (don't forget to disable the slider first!):

#input_data = [5.4,0.835,0.08,1.2,0.046,13.0,93.0,0.9924,3.57,0.85, 13.0]

input_df = pd.DataFrame([input_data],columns=feature_names)
with open('./model/scaler.pkl','rb') as file:
    scaler = pickle.load(file)

input_scaled = scaler.transform(input_df)

if st.button("### Predict Wine Quality"):
    prediction = model.predict(input_scaled)[0]
    
    proba = model.predict_proba(input_scaled)[0][1]

    st.write(f"### Prediction : {'Good(1)' if (proba >= 0.3) else 'Bad(0)'} wine")
    st.write(f"### Confidence : {proba * 100:.2f}%")
    

    st.caption("The model considers wines with a predicted probability > 30% as 'Good'.")

with st.expander("### Model Performance Summary"):
    st.markdown("""
    - **Accuracy:** ~89%
    - **Precision (Good):** 56%
    - **Recall (Good):** 81%
    - **F1 Score (Good):** 66%
    - **Class Imbalance:** 347 Bad vs 53 Good
    """)
