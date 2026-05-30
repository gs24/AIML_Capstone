
import streamlit as st

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference import predict

st.title("Engine Failure Prediction")

rpm = st.number_input("Engine RPM")
lub_oil_pressure = st.number_input("Lub Oil Pressure",step=1e-9, format="%.9f")
fuel_pressure = st.number_input("Fuel Pressure",step=1e-9, format="%.9f")
coolant_pressure = st.number_input("Coolant Pressure",step=1e-9, format="%.9f")
lub_oil_temp = st.number_input("Lub Oil Temp",step=1e-9, format="%.9f")
coolant_temp = st.number_input("Coolant Temp",step=1e-9, format="%.9f")

if st.button("Predict"):
    input_data = {
        "Engine rpm": rpm,
        "Lub oil pressure": lub_oil_pressure, 
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }

    
    prediction,model_name = predict(input_data)
    st.write(f"Model Used: {model_name}")
    if prediction == 1:
        st.error(f"Engine Failure May Occur!")
    else:
        st.success(f"Engines running healthily.")

        
