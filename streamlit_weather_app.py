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
    st.write(pivot_table)
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
elevation = st.number_input('Enter Elevation', value=0.0, format='%f')

# if st.button('Predict Pollution Using Radius Method'):
#     new_coordinates = np.array([[lat, lon, elevation]])
#     # Retrieve models from session state for prediction
#     prediction = st.session_state['pollutant_predictor'].predict_pollutants(new_coordinates)
#     st.write('Prediction from Pollutant Predictor:', prediction)
if st.button('Predict Pollution Using Stacking Method'):
    new_coordinates = np.array([[lat, lon, elevation]])
    # Implement prediction with stacking_predictor similarly
    evalutation = st.session_state['stacking_predictor'].predict_and_evaluate()
    st.write('Mean Absolute Error:', evalutation)
    #prediction = st.session_state['stacking_predictor'].predict_for_new_coordinates(new_coordinates)
    #st.write('Prediction from Stacking Predictor:', np.round(prediction))
#%%
#### adding sidebar





st.divider()


def apply_prediction(row):
    # Assuming model's predict method returns a list/array of predictions for each row
    data_for_prediction = np.array([[row['Elevation'], row['lat'], row['lon']]])
    if 'stacking_predictor' in st.session_state:
        predictions = st.session_state['stacking_predictor'].predict_for_new_coordinates(data_for_prediction)
        # Assuming predictions is now a list/array of predictions, one for each output
        return pd.Series(predictions)
    else:
        return pd.Series([np.nan, np.nan])  # Adjust the number of np.nan values based on the number of outputs



st.divider()
st.subheader('Uploading a file for predictions')
st.write('Please make sure the next coluumn are in the file - Elevation, lat, lon')


# Upload file
uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xlsx'])


# Allow the user to upload a file in the sidebar with specific file types

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]

    # Read the file based on its extension
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)

    # Ensure the DataFrame has the expected columns before proceeding
    expected_columns = ['Elevation', 'lat', 'lon']
    if all(column in df.columns for column in expected_columns):
        # Extract the necessary columns and convert to NumPy array for prediction
        new_data = df[expected_columns].to_numpy()

        # Check if the stacking_predictor model is available in session state
        if 'stacking_predictor' in st.session_state:
            # Use the model to predict on the new data
            prediction = st.session_state['stacking_predictor'].predict_for_new_coordinates(new_data)

            full_file = pd.concat([df[['Elevation', 'lat', 'lon']], prediction.drop(columns=['lat', 'lon'])], axis=1)
            st.write('full file', full_file)



            # Convert DataFrame to CSV for download
            csv = full_file.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
    )





        else:
            st.error('Model is not trained yet. Please train the model first.')
    else:
        st.error('Uploaded file does not contain the required columns.')
else:
    st.write("Please upload a file.")

