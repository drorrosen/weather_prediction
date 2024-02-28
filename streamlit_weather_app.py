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
    prediction = st.session_state['stacking_predictor'].predict_for_new_coordinates(new_coordinates)
    st.write('Prediction from Stacking Predictor:', np.round(prediction))
#%%
#### adding sidebar





st.divider()


# def apply_prediction(row):
#     # Assuming model's predict method returns a list/array of predictions for each row
#     data_for_prediction = np.array([[row['Elevation'], row['lat'], row['lon']]])
#     if 'stacking_predictor' in st.session_state:
#         predictions = st.session_state['stacking_predictor'].predict_for_new_coordinates(data_for_prediction)
#         # Assuming predictions is now a list/array of predictions, one for each output
#         return pd.Series(predictions)
#     else:
#         return pd.Series([np.nan, np.nan])  # Adjust the number of np.nan values based on the number of outputs

def apply_prediction(row):
    data_for_prediction = np.array([[row['Elevation'], row['lat'], row['lon']]])
    if 'stacking_predictor' in st.session_state:
        predictions = st.session_state['stacking_predictor'].predict_for_new_coordinates(data_for_prediction)

        # If predictions is a DataFrame, extract its values as numpy array
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        # Ensure predictions is in a format that can be converted to a Series directly
        # If predictions is a 2D numpy array with one row, flatten it to 1D
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            predictions = predictions.flatten()

        # Now, predictions should be in a format that can be directly converted to a Series
        return pd.Series(predictions)
    else:
        # Adjust the number of np.nan values based on the number of outputs expected
        return pd.Series([np.nan])  # Update 'number_of_expected_outputs' accordingly



# st.divider()
# st.subheader('Uploading a file for predictions')
# st.write('Please make sure the next coluumn are in the file - Elevation, lat, lon')
#
# uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xlsx'])
#
#
# # Allow the user to upload a file in the sidebar with specific file types
#
# if uploaded_file is not None:
#     file_extension = uploaded_file.name.split('.')[-1]
#     # Read the uploaded file
#     if file_extension == 'csv':
#         df = pd.read_csv(uploaded_file)
#     elif file_extension == 'xlsx':
#         df = pd.read_excel(uploaded_file)
#     else:
#         st.error('Invalid file type. Only CSV and Excel files are supported.')
#         st.stop()
#
#     # Apply the prediction function to each row
#     for index, row in df.iterrows():
#         prediction = apply_prediction(row)
#         st.write(prediction)

    # # Prepare data for prediction
    # prediction_input = df[expected_columns].to_numpy()
    #
    # if 'stacking_predictor' in st.session_state:
    #     # Obtain predictions for the entire dataset at once
    #     predictions = st.session_state['stacking_predictor'].predict_for_new_coordinates(prediction_input)
    #
    #     # If predictions is a DataFrame, you can directly concatenate; if it's a numpy array, convert it first
    #     if isinstance(predictions, np.ndarray):
    #         # Convert to DataFrame and name columns appropriately
    #         predictions_df = pd.DataFrame(predictions, columns=['Prediction1', 'Prediction2'])  # Adjust column names as needed
    #     else:
    #         # Assuming predictions is already a DataFrame with the correct column names
    #         predictions_df = predictions
    #         st.write('Predictions:', predictions_df)
    #
    #     # Ensure the index aligns with the original DataFrame if necessary
    #     predictions_df.reset_index(drop=True, inplace=True)
    #     predictions_df = predictions_df.drop(columns=['lat', 'lon'], errors='ignore')
    #
    #     df.reset_index(drop=True, inplace=True)
    #
    #     # Concatenate the predictions DataFrame with the original DataFrame
    #     full_df = pd.concat([df, predictions_df], axis=1)
    #
    #     st.write('Full file with predictions:', full_df)
    #
    #     # Convert to CSV for download
    #     csv = full_df.to_csv(index=False).encode('utf-8')
    #     st.download_button(
    #         label="Download predictions as CSV",
    #         data=csv,
    #         file_name='predictions.csv',
    #         mime='text/csv',
    #     )

# Upload file
