import streamlit as st
import pandas as pd
from Functions import *

def train_and_fit_models():
    # Function to train and fit models
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

# Button to train the model, only trains once if the model isn't already in the session state
if 'model' not in st.session_state or st.sidebar.button('Train Model'):
    st.session_state.model = train_and_fit_models()
    st.sidebar.success('Model Trained Successfully!')

with st.sidebar.form(key='client_info_form'):
    latitude = st.number_input("Latitude",  value=0.0, step=0.1)
    longitude = st.number_input("Longitude", value=0.0, step=0.1)
    elevation = st.number_input("Elevation", value=0.0, step=0.1)
    submit_button = st.form_submit_button(label='Run the pollution prediction')

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

    # Ensure model is trained
    if 'model' in st.session_state:
        # Predicting pollution
        prediction = predict_pollution(st.session_state.model, data_to_predict)

        # Display the prediction
        st.write("Predicted Pollution Levels:", prediction)
    else:
        st.error("Model not trained. Please train the model first.")
