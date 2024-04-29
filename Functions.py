import pandas as pd
import numpy as np
import requests
import json
from xgboost import XGBRegressor
from datetime import datetime
from scipy.spatial import cKDTree


import warnings
warnings.filterwarnings('ignore')

from scipy.stats import skew
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor as KNN
import pandas as pd

class DataFetcherAndPreprocessor:
    def __init__(self):
        self.base_url = "https://api.aqi.in/api/v1/getAllLocationsAqicn"
        self.raw_data = []
        self.flattened_data = []
        self.df = None

    def fetch_data(self):
        total_records = 15000  # Total records as mentioned
        records_per_request = 5000  # Max records we can fetch per request

        for offset in range(0, total_records, records_per_request):
            url = f"{self.base_url}?skip={offset}&take={records_per_request}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                self.raw_data.extend(data['Locations'])  # Adjust the key if different
            else:
                raise Exception(f"Failed to fetch data. Status code: {response.status_code}")



    def preprocess_data(self):
        if not self.raw_data:
            raise Exception("Data not fetched yet.")

        all_data = []

        for station in self.raw_data:
            # Current data handling
            for component in station['airComponents']:
                current_dict = {
                    'stationname': station['stationname'],
                    'locationId': station['locationId'],
                    'lat': station['lat'],
                    'lon': station['lon'],
                    'Elevation': station['Elevation'],
                    'updated_at': station['updated_at'],
                    'sensorName': component['sensorName'],
                    'sensorData': component['sensorData'],
                    'sensorUnit': component['sensorUnit'],
                    'timeframe': 'current'
                }
                all_data.append(current_dict)

            # Past data handling
            for past_record in station.get('pastdata', []):
                past_dict = {
                    'stationname': station['stationname'],
                    'locationId': station['locationId'],
                    'lat': station['lat'],
                    'lon': station['lon'],
                    'Elevation': station['Elevation'],
                    'updated_at': past_record['created_at'],
                    'sensorName': past_record['sensorname'],
                    'sensorData': float(past_record['sensorvalue']),
                    'sensorUnit': '',  # Assuming unit might not change, adjust if available
                    'timeframe': 'past'
                }
                all_data.append(past_dict)

        self.df = pd.DataFrame(all_data)
        self.df['updated_at'] = pd.to_datetime(self.df['updated_at'])


        print("Data preprocessing complete. DataFrame ready for analysis.")


    def create_pivot_table(self):
        self.df = self.df.drop('stationname', axis=1)
        # Convert updated_at to just date if it includes time
        self.df['date'] = pd.to_datetime(self.df['updated_at']).dt.date
        self.df = self.df.drop('updated_at', axis=1)




        # Pivot table creation
        pivot_table = self.df.pivot_table(
            index=['date', 'locationId', 'lat', 'lon', 'Elevation'],
            columns='sensorName',  # Columns will be created for each sensor type
            values='sensorData',  # The values to summarize
            aggfunc='median'  # Multiple aggregation functions can be applied
        )

        # Resetting index to turn multi-index into flat columns, which might be easier to handle
        pivot_table.reset_index(inplace=True)

        # Optionally, rename columns here if required
        #
        #only take last 7 days
        today = pd.Timestamp(datetime.today().date())
        pivot_table['date'] = pd.to_datetime(pivot_table['date'])
        pivot_table = pivot_table[(pivot_table['date'] > (today  - pd.DateOffset(days=7))) & (pivot_table['date'] <= today)]
        #pivot_table.columns = [' '.join(col).strip() for col in pivot_table.columns.values]

        return pivot_table


    def apply_qc_checks(self, df):
        pollutant_columns = [col for col in df.columns if col not in ['locationId', 'lat', 'lon', 'date','Elevation']]
        for col in pollutant_columns:
            df[col] = df[col].replace(999, np.nan)
        return df

    def drop_columns_with_missing_data(self, df, threshold=50):
        missing_percentage = df.isnull().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        df_cleaned = df.drop(columns=columns_to_drop)
        print(f"Columns dropped due to more than {threshold}% missing values: {columns_to_drop}")
        return df_cleaned


def fill_missing_dates(df_cleaned):
    # Define the date range for which new dates were potentially added
    date_range = pd.date_range(start=df_cleaned['date'].min(), end=df_cleaned['date'].max())

    # Create a reference table that contains constant attributes for each locationId
    reference_table = df_cleaned[['locationId', 'lat', 'lon', 'Elevation']].drop_duplicates('locationId')

    # Create a DataFrame with all combinations of 'locationId' and the date range
    all_combinations = pd.MultiIndex.from_product([df_cleaned['locationId'].unique(), date_range],
                                                  names=['locationId', 'date']).to_frame(index=False)

    # Merge the reference table with all_combinations to ensure constant attributes are replicated
    full_data = all_combinations.merge(reference_table, on='locationId', how='left')

    # Merge with the original data to find missing dates per location
    full_data = full_data.merge(df_cleaned.drop(columns=['lat', 'lon', 'Elevation']),
                                on=['locationId', 'date'], how='left', indicator=True)
    full_data.drop(columns=['_merge'], inplace=True)

    # Fill in missing data points by linear interpolation
    full_data = full_data.groupby('locationId').apply(lambda group: group.interpolate(method='linear')).reset_index(drop=True)

    return full_data





def impute_missing_values(full_data, predictors):
    # Define the target variables dynamically based on the DataFrame structure
    targets = full_data.drop(columns=['date', 'locationId'] + predictors).columns

    # Initialize the scaler and KNN imputer
    scaler = StandardScaler()
    imputer = KNN(n_neighbors=3)

    # Iterate over each unique date
    for date in full_data['date'].unique():
        date_df = full_data[full_data['date'] == date].copy()

        # Prepare predictors
        X = date_df[predictors]
        X_sc = scaler.fit_transform(X)  # Scale features

        # Iterate over each target variable
        for target in targets:
            y = date_df[target].copy()

            # Determine which data points are available vs missing
            available_data = y.notna()
            missing_data = y.isna()

            # Only proceed if there are any available data points
            if available_data.any():
                imputer.fit(X_sc[available_data], y[available_data])

                # Predict and impute missing values if there are any
                if missing_data.any():
                    y[missing_data] = imputer.predict(X_sc[missing_data])

                # Replace the original column data with the updated series
                full_data.loc[date_df.index, target] = y

    return full_data

# Example usage:
# predictors = ['lat', 'lon', 'Elevation']
# full_data = impute_missing_values(full_data, predictors)


# Usage
# df_cleaned = ... # your cleaned DataFrame
# stacking_predictor = StackingPollutantPredictor(df_cleaned)
# stacking_predictor.fit()
# predicted_df = stacking_predictor.predict_and_evaluate()
class StackingPollutantPredictor:
    def __init__(self, df):
        self.df_cleaned = df
        self.df_cleaned['Elevation'] = pd.to_numeric(self.df_cleaned['Elevation'], errors='coerce')
        self.X = self.df_cleaned[['lat', 'lon', 'Elevation', 'daylight_hours','closest_city_km']]
        self.Y = self.df_cleaned.drop(columns=['lat', 'lon', 'Elevation', 'daylight_hours','closest_city_km'])

        self.models = [
            RandomForestRegressor(random_state=42),
            KNeighborsRegressor(n_neighbors=3),
            GradientBoostingRegressor(random_state=42),
            XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
        ]
        self.stacker = XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)

    def fit(self, test_size=0.1, random_state=42, n_splits=5):


        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        n_outputs = Y_train.shape[1]
        oof_preds = np.zeros((X_train.shape[0], n_outputs * len(self.models)))

        for train_idx, valid_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            Y_tr, Y_val = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

            for i, model in enumerate(self.models):
                if not hasattr(model, "predict_proba") and model._estimator_type == "regressor":
                    model = MultiOutputRegressor(model)
                model_clone = clone(model)
                model_clone.fit(X_tr, Y_tr)

                preds = model_clone.predict(X_val)
                oof_preds[valid_idx, i*n_outputs:(i+1)*n_outputs] = preds

        self.stacker.fit(oof_preds, Y_train)
        self.X_test, self.Y_test = X_test, Y_test  # Store for evaluation

    def predict_for_new_coordinates(self, new_coordinates):
        """
        Predict pollutant levels for new coordinates using the trained stacking model.

        Parameters:
        - new_coordinates: A 2D array-like or list of lists containing latitude and longitude pairs.

        Returns:
        - A DataFrame containing the predicted pollutant levels for the new coordinates.
        """
        # Ensure new_coordinates is in the correct format
        if not isinstance(new_coordinates, np.ndarray):
            new_coordinates = np.array(new_coordinates)

        # Generate base model predictions for the new coordinates
        n_outputs = self.Y.shape[1]
        new_preds = np.zeros((new_coordinates.shape[0], n_outputs * len(self.models)))

        for i, model in enumerate(self.models):
            # Check if the model needs to be wrapped with MultiOutputRegressor
            if not hasattr(model, "predict_proba") and model._estimator_type == "regressor":
                model = MultiOutputRegressor(model)
            # It's assumed that the models are already fitted; if not, you might need to fit them again here

            # Predict with the base model
            model.fit(self.X, self.Y)
            preds = model.predict(new_coordinates)
            new_preds[:, i*n_outputs:(i+1)*n_outputs] = preds

        # Use the stacking model to make the final predictions
        final_predictions = self.stacker.predict(new_preds)

        # Create a DataFrame for the predictions
        predicted_df = pd.DataFrame(final_predictions, columns=self.Y.columns)
        predicted_df['lat'], predicted_df['lon'], predicted_df['Elevation'], predicted_df['daylight_hours'], predicted_df['closest_city_km'] = new_coordinates[:, 0], new_coordinates[:, 1], new_coordinates[:, 2], new_coordinates[:, 3], new_coordinates[:, 4]

        return predicted_df



    def save_model(self, filename='stacking_pollutant_predictor.joblib'):
        """Saves the model to a file."""
        dump(self, filename)

    @staticmethod
    def load_model(filename='stacking_pollutant_predictor.joblib'):
        """Loads the model from a file."""
        return load(filename)
    def predict_and_evaluate(self):
        n_outputs = self.Y_test.shape[1]
        test_preds = np.zeros((self.X_test.shape[0], n_outputs * len(self.models)))

        for i, model in enumerate(self.models):
            model_clone = clone(model)
            if not hasattr(model, "predict_proba") and model._estimator_type == "regressor":
                model_clone = MultiOutputRegressor(model_clone)
            model_clone.fit(self.X, self.Y)

            preds = model_clone.predict(self.X_test)
            test_preds[:, i*n_outputs:(i+1)*n_outputs] = preds

        final_predictions = self.stacker.predict(test_preds)
        final_mae = mean_absolute_error(self.Y_test, final_predictions)


        # Example modification to include after final_mae calculation
        maes_per_pollutant = {}
        for i, pollutant in enumerate(self.Y_test.columns):
            pollutant_mae = mean_absolute_error(self.Y_test.iloc[:, i], final_predictions[:, i])
            maes_per_pollutant[pollutant] = pollutant_mae
            #print(f"MAE for {pollutant}: {pollutant_mae}")

        # pd.DataFrame(final_predictions, columns=self.Y_test.columns)
        return final_mae, maes_per_pollutant


def train_models():
    fetcher = DataFetcherAndPreprocessor()
    fetcher.fetch_data()
    fetcher.preprocess_data()
    pivot_table = fetcher.create_pivot_table()
    # pivot_table = fetcher.filter_and_pivot_data()
    pivot_table_qc = fetcher.apply_qc_checks(pivot_table.copy())
    #pollutants = pivot_table_qc.drop(columns=['date','locationId', 'lat', 'lon']).columns
    # pivot_table_imputed = fetcher.impute_missing_values(pivot_table_qc, pollutants)
    pivot_table_qc = pivot_table_qc.sort_values(['locationId', 'date'], ascending=True)
    #pivot_table_qc = pivot_table_qc.groupby('locationId').apply(lambda group: group.ffill()).reset_index(drop=True)
    df_cleaned = fetcher.drop_columns_with_missing_data(pivot_table_qc)
    df_cleaned= df_cleaned.drop(columns = ['AQI-IN', 'aqi'], axis = 1)
    full_data = fill_missing_dates(df_cleaned)
    predictors = ['lat', 'lon', 'Elevation']
    full_data = impute_missing_values(full_data, predictors)
    full_data = full_data.loc[full_data['date'] == full_data['date'].max()]
    full_data = full_data.drop(columns=['locationId','date'])
    full_data = full_data.astype(float)
    full_data['daylight_hours'] = full_data['lat'].apply(lambda x: daylight_hours(x))

    full_data = remove_outliers(full_data)
    full_data['closest_city_km'] = calculate_distances_to_cities( full_data[['lat', 'lon']].values)
    return full_data

def solar_declination():
    day_of_year = (datetime.today()).timetuple().tm_yday
    return 23.45 * np.sin(np.radians((360 / 365) * (day_of_year + 10)))

def daylight_hours(latitude):
    latitude_rad = np.radians(latitude)
    declination_rad = np.radians(solar_declination())
    cos_omega = (np.sin(np.radians(-0.83)) - np.sin(latitude_rad) * np.sin(declination_rad)) / (np.cos(latitude_rad) * np.cos(declination_rad))
    cos_omega = np.clip(cos_omega, -1, 1)  # Ensure the value stays within the valid range of arccos
    omega = np.arccos(cos_omega)
    return (24 / np.pi) * omega



def remove_outliers(full_data):
    """
    Removes outliers from a DataFrame based on skewness.

    Parameters:
    - full_data: pandas DataFrame containing the data.
    - columns_to_ignore: list of column names to exclude from outlier detection.

    Returns:
    - A tuple with the cleaned DataFrame and the number of removed outliers.
    """
    # Initialize a set to store indices of outliers across all pollutants
    outlier_indices = set()

    # Iterate through each pollutant
    for pollutant in full_data.drop(columns=['lat', 'lon', 'Elevation', 'daylight_hours']).columns:
        # Calculate skewness of the pollutant
        pollutant_skewness = skew(full_data[pollutant].dropna())  # Drop NA to calculate skewness accurately

        # Check if the absolute value of skewness is greater than 1
        if abs(pollutant_skewness) > 1:
            if pollutant_skewness > 1:
                # Skewness is positive, remove high outliers
                quantile_high = full_data[pollutant].quantile(0.99)
                high_outliers = full_data[full_data[pollutant] > quantile_high].index
                outlier_indices.update(high_outliers)
            elif pollutant_skewness < -1:
                # Skewness is negative, remove low outliers
                quantile_low = full_data[pollutant].quantile(0.01)
                low_outliers = full_data[full_data[pollutant] < quantile_low].index
                outlier_indices.update(low_outliers)


    # Remove all rows that have outliers in any pollutant
    cleaned_data = full_data.drop(index=list(outlier_indices))

    return cleaned_data


def calculate_distances_to_cities(full_data_coords):

    data_file_path = 'World_Cities_Location_table_MS-EXCEL.csv'

    # Set the column names expected in the CSV file
    column_names = ['id', 'country', 'city', 'lat', 'lon', 'Elevation']

    # Load city data from the file with corrected headers
    city_data = pd.read_csv(data_file_path, header=None, skiprows=1, names=column_names)

    # Extract the latitude and longitude coordinates for cities
    cities_coords = city_data[['lat', 'lon']].values

    # Create a k-d tree for efficient distance calculation
    tree = cKDTree(cities_coords)

    # Earth's radius in kilometers for distance conversion
    earth_radius_km = 6371.0

    # Calculate the closest city for each coordinate in full_data_coords
    distance, index = tree.query(full_data_coords, k=1)

    # Convert distance from degrees to kilometers
    distances_km = distance * np.pi / 180 * earth_radius_km

    # Return the distances in kilometers
    return distances_km


