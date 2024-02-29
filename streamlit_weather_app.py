import streamlit as st
from Functions import *
import numpy as np
import pandas as pd
import joblib


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
    # Assuming `df_cleaned` is your cleaned DataFrame ready for training
    stacking_predictor.save_model('./my_trained_model.joblib')  # Save your model to a file

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


if st.button('Predict Pollution Using Stacking Method'):
    new_coordinates = np.array([[lat, lon, elevation]])
    st.write('New Coordinates:', new_coordinates)
    # Implement prediction with stacking_predictor similarly
    evalutation = st.session_state['stacking_predictor'].predict_and_evaluate()
    st.write('Mean Absolute Error:', evalutation)
    prediction = st.session_state['stacking_predictor'].predict_for_new_coordinates(new_coordinates)
    st.write('Prediction from Stacking Predictor:', np.round(prediction))
#%%
#### adding sidebar






st.divider()
st.subheader('JSON input')
# Load the trained model
model = joblib.load('./my_trained_model.joblib')

# Title of the web app
st.title('Pollutant Level Prediction')

# User inputs JSON in the text area
example = """[
  {"Elevation": 100, "lat": 30.1204, "lon": 74.29},
  {"Elevation": 200, "lat": 24.5925, "lon": 72.7083},
  {"Elevation": 300, "lat": 15.811, "lon": 79.9738}
]"""
user_input_json = st.text_area("Enter coordinates in JSON format", example)

# Button to make predictions
if st.button('Predict'):
    try:
        # Parse the user input to JSON
        new_coordinates = json.loads(user_input_json)

        # Assuming your model expects a 2D array-like structure for new coordinates
        # Convert list of dicts to the expected format [[lat, lon], [lat, lon], ...]
        # Convert list of dicts to the expected format [[lat, lon, Elevation], ...]
        formatted_coordinates = [[coord['lat'], coord['lon'], coord['Elevation']] for coord in new_coordinates]



        # Now new_coordinates is ready for model prediction
        predictions = model.predict_for_new_coordinates(formatted_coordinates)

        # Convert predictions to a DataFrame for nicer display
        predictions_df = pd.DataFrame(predictions, columns=model.Y.columns) # Adjust columns as per your model's output
        predictions_df['Elevation'] = [coord['Elevation'] for coord in new_coordinates]
        predictions_df['lat'] = [coord['lat'] for coord in new_coordinates]
        predictions_df['lon'] = [coord['lon'] for coord in new_coordinates]

        #round the results
        predictions_df = predictions_df.round()
        # Display predictions
        st.write(predictions_df)

        #download button to download the prediction
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )


    except json.JSONDecodeError:
        st.error("There was an error decoding the JSON. Please check the input format.")
    except Exception as e:
        st.error(f"An error occurred: {e}")





