# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:50:21 2025

@author: User
"""

import streamlit as st
import pickle
import pandas as pd

# Load saved model and encoders
model = pickle.load(open('loan_approval_model.pkl', 'rb'))
status_encoder = pickle.load(open('employment_status_encoder.pkl', 'rb'))
approval_encoder = pickle.load(open('approval_encoder.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Prediction App")
st.header("Project 2 - Group 9")
st.subheader("Enter Customer Details:")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    income = st.number_input('Annual Income ($)', min_value=0, value=50000)

with col2:
    credit_score = st.slider('Credit Score', 300, 850, 700)

with col1:
    loan_amount = st.number_input('Loan Amount ($)', min_value=0, value=10000)

with col2:
    dti_ratio = st.slider('Debt-to-Income Ratio (%)', 0.0, 100.0, 30.0)

# Dynamically use the classes from the loaded encoder
employment_status = st.selectbox('Employment Status', status_encoder.classes_)

# Predict button
if st.button('Predict Approval'):
    with st.spinner('Predicting...'):
        # Prepare input
        employment_encoded = status_encoder.transform([employment_status])[0]
        
        input_data = pd.DataFrame([[
            income,
            credit_score,
            loan_amount,
            dti_ratio,
            employment_encoded
        ]], columns=['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status'])

        # Predict
        prediction = model.predict(input_data)
        prediction_label = approval_encoder.inverse_transform(prediction)[0]

    # Display result
    if prediction_label == 'Yes':
        st.success("🎉 Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected!")



