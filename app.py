import streamlit as st
import joblib
import numpy as np

# Load the trained model
with open("result.pkl", "rb") as f:
    model = joblib.load(f)

# Streamlit UI
st.title("House Price Prediction")

# User inputs
gr_liv_area = st.number_input("Above ground living area (GrLivArea)", min_value=0, value=1500)
bedroom_abv_gr = st.number_input("Number of bedrooms above ground (BedroomAbvGr)", min_value=0, value=3)
full_bath = st.number_input("Number of full bathrooms (FullBath)", min_value=0, value=2)

# Prediction
if st.button("Predict Price"):
    features = np.array([[gr_liv_area, bedroom_abv_gr, full_bath]])
    prediction = model.predict(features)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
