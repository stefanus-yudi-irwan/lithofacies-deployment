import streamlit as st
import requests
import joblib
from PIL import Image

# Load and set images in the first place
header_images = Image.open('image/oil_well_banner.jpg')
st.image(header_images)

# Add some information about the service
st.title("Lithofacies Classification")
st.subheader("Enter measurement value then click predict button")

# Create form of input
with st.form(key = "well_measurement_form"):
    # Create box for number input
    GR = st.number_input(
        "1.\tEnter Gamma Ray Value:",
        min_value = 10.14,
        max_value = 361.15,
        help = "Value range from 10.14 to 361.15"
    )

    ILD_log10 = st.number_input(
        "2.\tEnter Resistivity Value:",
        min_value = -0.03,
        max_value = 1.8,
        help = "Value range from -0.03 to 1.8"
    )

    DeltaPHI = st.number_input(
        "3.\tEnter Neutron-Density Porosity Difference Value:",
        min_value = -21.83,
        max_value = 19.31,
        help = "Value range from -21.83 to 19.31"
    )

    PHIND = st.number_input(
        "4.\tEnter Average Neutron-Density Porosity Value:",
        min_value = 0.55,
        max_value = 84.4,
        help = "Value range from 0.55 to 84.4"
    )

    PE = st.number_input(
        "5.\tEnter Photo-Electric Value:",
        min_value = 0.2,
        max_value = 8.09,
        help = "Value range from 0.2 to 8.09"
    )

    NM_M = st.number_input(
        "6.\tEnter Non-Marine Marine Value:",
        min_value = 1,
        max_value = 2,
        help = "Value 1 or 2"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            'GR' : GR,
            'ILD_log10' : ILD_log10,
            'DeltaPHI' : DeltaPHI,
            'PHIND' : PHIND,
            'PE' : PE,
            'NM_M' : NM_M,
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api_backend:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            st.success(res["res"])