import streamlit as st
from Functions import *
import numpy as np
import pandas as pd


# Function to train and cache models
@st.cache_data
def train_models():
    # Initialize and preprocess data
    fetcher = DataFetcherAndPreprocessor()
    fetcher.fetch_data()
    fetcher.preprocess_data()
    fetcher.extract_and_process_dates()
    pivot_table = fetcher.filter_and_pivot_data()
    pivot_table_qc = fetcher.apply_qc_checks(pivot_table.copy())
    pollutants = pivot_table_qc.drop(columns=['locationId', 'lat', 'lon']).columns
    pivot_table_imputed = fetcher.impute_missing_values(pivot_table_qc, pollutants)
    df_cleaned = fetcher.drop_columns_with_missing_data(pivot_table_imputed)
    df_cleaned.set_index('locationId', inplace=True)
    df_cleaned = df_cleaned.dropna()
    # Train predictors
    pollutant_predictor = PollutantPredictor(df_cleaned)
    stacking_predictor = StackingPollutantPredictor(df_cleaned)
    stacking_predictor.fit()

    return pollutant_predictor, stacking_predictor

st.title("Pollution Prediction App")

# Check if models are already trained and stored in session state
if 'pollutant_predictor' not in st.session_state or 'stacking_predictor' not in st.session_state:
    if st.button('Train the Models'):
        pollutant_predictor, stacking_predictor = train_models()
        # Store models in session state
        st.session_state['pollutant_predictor'] = pollutant_predictor
        st.session_state['stacking_predictor'] = stacking_predictor
        st.success('Models trained successfully!')
else:
    st.success('Models are already trained!')

lat = st.number_input('Enter Latitude', value=0.0, format='%f')
lon = st.number_input('Enter Longitude', value=0.0, format='%f')

if st.button('Predict Pollution Using Radius Method'):
    new_coordinates = np.array([[lat, lon]])
    # Retrieve models from session state for prediction
    prediction = st.session_state['pollutant_predictor'].predict_pollutants(new_coordinates)
    st.write('Prediction from Pollutant Predictor:', prediction)
if st.button('Predict Pollution Using Stacking Method'):
    new_coordinates = np.array([[lat, lon]])
    # Implement prediction with stacking_predictor similarly
    prediction = st.session_state['stacking_predictor'].predict_for_new_coordinates(new_coordinates)
    st.write('Prediction from Stacking Predictor:', np.round(prediction))
#%%
