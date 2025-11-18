import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.utils import load_ann_model,load_preprocessor
import os

preprocessor_path=os.path.join("artifacts","regression","preprocessor.pkl")
preprocessor=load_preprocessor(preprocessor_path)

model_path=os.path.join("artifacts","regression","model.h5")
model=load_ann_model(model_path)

## streamlit app
st.title('Customer Churn Prediction')

geography_cat=['France','Germany','Spain']
gender_cat=['Male','Female']

# User input
geography = st.selectbox('Geography', geography_cat)
gender = st.selectbox('Gender', gender_cat)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = 0 if st.radio("Exited",["Yes","No"],index=0,)=='Yes' else 1

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography':[geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

df=preprocessor.transform(input_data)
prediction=model.predict(df)
prediction_proba=prediction[0][0]

st.write(f'Expected Salary: {prediction_proba:.2f}')