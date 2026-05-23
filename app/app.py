import streamlit as st

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from deployment.inference import predict

st.title("Engine Failure Prediction")

rpm = st.number_input("Engine RPM")
lub_oil_pressure = st.number_input("Lub Oil Pressure")
fuel_pressure = st.number_input("Fuel Pressure")
coolant_pressure = st.number_input("Coolant Pressure")
lub_oil_temp = st.number_input("Lub Oil Temp")
coolant_temp = st.number_input("Coolant Temp")

if st.button("Predict"):
    input_data = {
        "Engine rpm": rpm,
        "Lub oil pressure": lub_oil_pressure, 
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }

    
    prediction = predict(input_data)
    
    if prediction == 1:
        st.error(f"Engine Failure May Occur!")
    else:
        st.success(f"Engines running healthily.")

    