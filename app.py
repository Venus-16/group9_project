# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:50:21 2025

@author: User
"""

import streamlit as st
import pandas as pd
import pickle
import time

# Load model and encoders
model = pickle.load(open('loan_approval_model.pkl', 'rb'))
status_encoder = pickle.load(open('employment_encoder.pkl', 'rb'))
approval_encoder = pickle.load(open('approval_encoder.pkl', 'rb'))

# Page title
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Prediction App")
st.header("Project 2 - Group 9")
st.subheader("Enter Customer Details:")

# Input form
income = st.number_input('Annual Income (USD)', min_value=0)
credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
loan_amount = st.number_input('Loan Amount (USD)', min_value=0)
dti_ratio = st.number_input('Debt-to-Income Ratio (%)', min_value=0.0, max_value=100.0)
employment_status = st.selectbox('Employment Status', status_encoder.classes_)

# Predict button
# Predict button
if st.button('Predict Loan Approval'):
    with st.spinner('🔎 Analyzing customer profile...'):
        time.sleep(2)

        input_data = pd.DataFrame({
            'Income': [income],
            'Credit_Score': [credit_score],
            'Loan_Amount': [loan_amount],
            'DTI_Ratio': [dti_ratio],
            'Employment_Status': [status_encoder.transform([employment_status])[0]]
        })

        prediction = model.predict(input_data)
        result = approval_encoder.inverse_transform(prediction)[0]

    # Display result with different color
    if result == 'Approved':
        st.markdown(
            "<h4 style='text-align: center; color: green;'>✅ Loan Approved!</h4>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h4 style='text-align: center; color: red;'>❌ Loan Not Approved</h4>",
            unsafe_allow_html=True
        )



