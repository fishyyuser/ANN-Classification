import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.utils import load_ann_model,load_preprocessor
import os

preprocessor_path=os.path.join("artifacts","classification","preprocessor.pkl")
preprocessor=load_preprocessor(preprocessor_path)

model_path=os.path.join("artifacts","classification","model.h5")
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
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

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
    'EstimatedSalary': [estimated_salary]
})

df=preprocessor.transform(input_data)
prediction=model.predict(df)
prediction_proba=prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')