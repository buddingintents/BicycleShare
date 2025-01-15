import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# Load the model training function from the uploaded file
# Assuming this is similar to your provided logic

# Define a function to train the model and return R² and RMSE for training data
def train_model(train_data):
    try:
        # Ensure proper data types and handle date columns
        train_data = train_data.copy()
        if 'dteday' in train_data.columns:
            train_data.drop('dteday', axis=1, inplace=True)  # Remove date column if present

        y = train_data['cnt']
        x = train_data.drop(['cnt', 'atemp', 'registered'], axis=1)  # Drop multicollinear columns

        # Train model
        reg_model = LinearRegression().fit(x, y)
        r_squared = reg_model.score(x, y)
        predictions = reg_model.predict(x)
        train_data['predicted_cnt'] = predictions  # Add predictions to the training data
        rmse = math.sqrt(mean_squared_error(y, predictions))
        return reg_model, r_squared, rmse, train_data
    except Exception as e:
        st.error(f"Error in training model: {str(e)}")
        return None, None, None, None

# Define a function to evaluate the model on test data
def evaluate_model(model, test_data):
    try:
        if 'dteday' in test_data.columns:
            test_data.drop('dteday', axis=1, inplace=True)  # Remove date column if present

        y_test = test_data['cnt']
        x_test = test_data.drop(['cnt', 'atemp', 'registered'], axis=1)

        predictions = model.predict(x_test)
        test_data['predicted_cnt'] = predictions  # Add predictions to the test data
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        return rmse, test_data
    except Exception as e:
        st.error(f"Error in evaluating model: {str(e)}")
        return None, None

# Streamlit App Interface
st.set_page_config(page_title="Ankit's App")
st.title("Bike Sharing Demand Prediction Model")

model = None  # Initialize model variable

# Upload train data
train_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
if train_file is not None:
    train_data = pd.read_csv(train_file)
    st.write("Training Data Sample:", train_data.head())

    model, r_squared, train_rmse, train_data_with_predictions = train_model(train_data)
    if model:
        st.write(f"R² score on training data: {r_squared:.4f}")
        st.write(f"RMSE on training data: {train_rmse:.4f}")
        st.write("Training Data with Predictions:", train_data_with_predictions.head())

# Upload test data
if model:
    test_file = st.file_uploader("Upload Test Data (CSV)", type="csv")
    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data Sample:", test_data.head())

        test_rmse, test_data_with_predictions = evaluate_model(model, test_data)
        if test_rmse:
            st.write(f"RMSE on test data: {test_rmse:.4f}")
            st.write("Test Data with Predictions:", test_data_with_predictions.head())
