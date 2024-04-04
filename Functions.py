import pandas as pd
import numpy as np
import requests
import json
from joblib import dump, load
from xgboost import XGBRegressor



import warnings
warnings.filterwarnings('ignore')


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



class DataFetcherAndPreprocessor:
    def __init__(self):
        self.url = "https://api.aqi.in/api/v1/getAllLocationsAqicn?skip=0&take=15000"
        self.raw_data = None
        self.flattened_data = []
        self.df = None

    def fetch_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            self.raw_data = json.loads(json.dumps(response.json()))
        else:
            raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    def preprocess_data(self):
        if not self.raw_data:
            raise Exception("Data not fetched yet.")

        for location in self.raw_data['Locations']:
            for component in location['airComponents']:
                flat_dict = {**location, **component}
                flat_dict.pop('airComponents', None)
                self.flattened_data.append(flat_dict)

        self.df = pd.DataFrame(self.flattened_data)

    def extract_and_process_dates(self):
        if self.df is None:
            raise Exception("Data not preprocessed yet.")

        self.df['date'] = pd.to_datetime(self.df['updated_at']).dt.date
        self.df['date'] = pd.to_datetime(self.df['date'])

    def filter_and_pivot_data(self, days=3):
        if self.df is None:
            raise Exception("Data not preprocessed yet.")

        latest_date = self.df['date'].max()
        filtered_df = self.df[self.df['date'] > latest_date - pd.Timedelta(days=days)]
        pivot_table = filtered_df.pivot_table(index=['locationId', 'lat', 'lon'],
                                              columns='sensorName',
                                              values='sensorData',
                                              aggfunc='mean').reset_index()
        pivot_table = pivot_table.drop(columns=['AQI-IN','aqi'])
        #pivot_table['Elevation'] = pivot_table['Elevation'].astype(int)
        return pivot_table

    def apply_qc_checks(self, df):
        pollutant_columns = [col for col in df.columns if col not in ['locationId', 'lat', 'lon']]
        for col in pollutant_columns:
            df[col] = df[col].replace(999, np.nan)
        return df

    def impute_missing_values(self, df, pollutants, radius=0.5):
        X = df[['lat', 'lon']]
        for pollutant in pollutants:
            y = df[pollutant]
            available_data = y.notna()
            if available_data.any():
                regressor = RadiusNeighborsRegressor(radius=radius)
                regressor.fit(X[available_data], y[available_data])
                missing_data = y.isna()
                if missing_data.any():
                    imputed_values = regressor.predict(X[missing_data])
                    df.loc[missing_data, pollutant] = imputed_values
        df['lat'] = df['lat'].astype(float)
        df['lon'] = df['lon'].astype(float)
        return df


    def drop_columns_with_missing_data(self, df, threshold=30):
        missing_percentage = df.isnull().mean() * 100
        columns_to_drop = missing_percentage[missing_percentage > threshold].index
        df_cleaned = df.drop(columns=columns_to_drop)
        print(f"Columns dropped due to more than {threshold}% missing values: {columns_to_drop}")
        return df_cleaned



class PollutantPredictor:
    def __init__(self, df_cleaned):
        self.df_cleaned = df_cleaned
        self.X = df_cleaned[['lat', 'lon']]  # Features
        self.Y = df_cleaned.drop(columns=['lat', 'lon'])  # Targets

    def predict_pollutants(self, new_coordinates, radii=np.arange(0, 2.1, 0.1)):
        # Initialize placeholders for predictions and the successful radius
        predictions = np.full((len(new_coordinates), self.Y.shape[1]), np.nan)
        successful_radii = np.full(len(new_coordinates), np.nan)

        # Iterate over the sequence of radii, attempting predictions
        for radius in radii:
            regressor = RadiusNeighborsRegressor(radius=radius)
            regressor.fit(self.X, self.Y)

            temp_predictions = regressor.predict(new_coordinates)

            # Update predictions and successful radii for previously NaN predictions
            for i, pred in enumerate(temp_predictions):
                if np.isnan(predictions[i]).any():  # Check if the prediction contains NaN
                    predictions[i] = pred
                    successful_radii[i] = radius

                    # Break if all predictions are filled for this radius
                    if not np.isnan(predictions).any():
                        break





        # Convert predictions to a DataFrame for easier interpretation
        predicted_df = pd.DataFrame(predictions, columns=self.Y.columns)
        predicted_df['lat'], predicted_df['lon'] = new_coordinates[:, 0], new_coordinates[:, 1]
        predicted_df['Successful Radius'] = successful_radii  # Add the successful radius to the DataFrame


        # # Apply the conversion from radius in degrees to kilometers
        predicted_df['radius_in_km'] = self.radius_to_km(predicted_df)

        return predicted_df


    @staticmethod
    def radius_to_km(df):
        # Function to convert radius from degrees to kilometers
        df['radius_in_km'] = df.apply(lambda row: (
                ((row['Successful Radius'] * 111) +
                 (row['Successful Radius'] * 111 * np.cos(row['lat'] * np.pi / 180))) / 2
        ), axis=1)
        return df['radius_in_km']






# Usage
# df_cleaned = ... # your cleaned DataFrame
# stacking_predictor = StackingPollutantPredictor(df_cleaned)
# stacking_predictor.fit()
# predicted_df = stacking_predictor.predict_and_evaluate()
class StackingPollutantPredictor:
    def __init__(self, df_cleaned):
        self.df_cleaned = df_cleaned
        #self.df_cleaned['Elevation'] = pd.to_numeric(self.df_cleaned['Elevation'], errors='coerce')
        self.X = df_cleaned[['lat', 'lon']]
        self.Y = df_cleaned.drop(columns=['lat', 'lon'])

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
        predicted_df['lat'], predicted_df['lon'] = new_coordinates[:, 0], new_coordinates[:, 1]

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
    #pollutant_predictor = PollutantPredictor(df_cleaned)
    stacking_predictor = StackingPollutantPredictor(df_cleaned)
    stacking_predictor.fit()
    # Assuming `df_cleaned` is your cleaned DataFrame ready for training
    #stacking_predictor.save_model('my_trained_model.joblib')  # Save your model to a file

    return stacking_predictor, df_cleaned.columns
