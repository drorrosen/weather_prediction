import streamlit as st
from Functions import *
import numpy as np
import pandas as pd
import joblib


# Function to train and cache models

st.title("Pollution Prediction App")



if st.button('Predict Pollutant Levels'):
    data_to_predict = pd.read_csv("GlobalLocations.csv")

    stacking_predictor, columns_to_add = train_models()


    # Extract latitude and longitude as a 2D numpy array
    coordinates = data_to_predict[['latitude', 'longitude']].values

    # Get predictions for all coordinates at once
    predictions_df = stacking_predictor.predict_for_new_coordinates(coordinates)
    predictions_df['id'] = data_to_predict['id']

    st.write(predictions_df)


    #download button to download the prediction
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
        )







