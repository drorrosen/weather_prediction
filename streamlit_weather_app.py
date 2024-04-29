import streamlit as st
import pandas as pd
from Functions import *
@st.cache_data()
def train_and_fit_models():
    full_data = train_models()
    stacking_predictor = StackingPollutantPredictor(full_data)
    stacking_predictor.fit()
    return stacking_predictor

# Function to perform prediction using the trained model
def predict_pollution(model, data):
    # Assuming the model has a method `predict` which can be used here
    return model.predict_for_new_coordinates(data)

# Main app setup
st.title("Pollution Prediction App")
st.sidebar.header("User Input Coordinates")

with st.sidebar.form(key='client_info_form'):
    latitude = st.number_input("Latitude",  value=0.0, step=0.1)
    longitude = st.number_input("Longitude",value=0.0, step=0.1)
    elevation = st.number_input("Elevation",value=0.0, step=0.1)
    submit_button = st.form_submit_button(label='Run the weather prediction')

if submit_button:
    # Displaying input data
    st.write(f'Latitude: {latitude}')
    st.write(f'Longitude: {longitude}')
    st.write(f'Elevation: {elevation}')

    # Preparing data for prediction
    data_to_predict = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'elevation': [elevation],
        'daylight_hours': [daylight_hours(latitude)],
        'closest_city_km': [calculate_distances_to_cities([[latitude, longitude]])]
    })

    # Getting the cached model
    model = train_and_fit_models()

    # Predicting pollution
    prediction = predict_pollution(model, data_to_predict)

    # Display the prediction
    st.write("Predicted Pollution Levels:", prediction)
