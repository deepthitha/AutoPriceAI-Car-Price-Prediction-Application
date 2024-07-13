import streamlit as st
import pandas as pd
import subprocess
import sys
import pickle
import mysql.connector
from datetime import datetime
import requests
import json

# Initialize session state
if 'model_needs_refresh' not in st.session_state:
    st.session_state.model_needs_refresh = False

# Load model, label encoders, and feature names
@st.cache(allow_output_mutation=True)
def load_assets():
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, label_encoders, feature_names

# Function to preprocess user input and make prediction
def predict_price(model, label_encoders, feature_names, user_input):
    # Encode categorical variables
    for key, value in user_input.items():
        if key in label_encoders:
            user_input[key] = label_encoders[key].transform([value])[0]
        elif key in ['engine_has_gas', 'has_warranty']:  # Encode boolean strings
            user_input[key] = 1 if value == 'TRUE' else 0

    # Fill in the remaining features with default values (e.g., mean or most frequent value)
    for feature in feature_names:
        if feature not in user_input:
            user_input[feature] = 0  # Replace with an appropriate default value for each feature

    # Create feature array in the correct order
    features = [user_input[feature] for feature in feature_names]

    # Predict price using the loaded model
    predicted_price = model.predict([features])[0]
    return predicted_price

# Function to connect to MySQL database
def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',  # Replace with your MySQL username
            password='2003',  # Replace with your MySQL password
            database='car_data'  # Replace with your MySQL database name
        )
        if conn.is_connected():
            return conn
        else:
            st.error('Database connection error!')
            return None
    except mysql.connector.Error as e:
        st.error(f'Error connecting to MySQL: {e}')
        return None

# Function to insert data into MySQL
def insert_data_to_mysql(conn, df):
    try:
        cursor = conn.cursor()
        # Prepare INSERT statement
        insert_query = """
        INSERT INTO car_listings
        (manufacturer_name, model_name, transmission, color, odometer_value, year_produced,
         engine_fuel, engine_has_gas, engine_type, engine_capacity, body_type, has_warranty,
         state, drivetrain, price_usd, is_exchangeable, location_region, number_of_photos,
         up_counter, feature_0, feature_1, feature_2, feature_3, feature_4, feature_5,
         feature_6, feature_7, feature_8, feature_9, duration_listed)
        VALUES
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Convert DataFrame to list of tuples for insertion
        records_to_insert = df.values.tolist()
        # Execute INSERT query
        cursor.executemany(insert_query, records_to_insert)
        # Commit changes
        conn.commit()
        st.success('Data inserted successfully into MySQL.')
        st.session_state.model_needs_refresh = True  # Set the flag to indicate model needs refresh

    except mysql.connector.Error as e:
        st.error(f'Error inserting data: {e}')

# Function to write the last refresh date to a file
def write_last_refresh_date():
    with open('last_refresh.txt', 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Function to read the last refresh date from the file
def read_last_refresh_date():
    try:
        with open('last_refresh.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return 'Never'

# Function to run model retraining
def run_model_retraining():
    try:
        st.write('Triggering model retraining...')
        python_path = sys.executable
        result = subprocess.run([python_path, 'train.py'], capture_output=True, text=True, check=True)
        if result.returncode == 0:
            st.success('Model retraining complete.')
            write_last_refresh_date()  # Update the last refresh date
            st.session_state.model_needs_refresh = False  # Reset the flag after model refresh
        else:
            st.error(f'Error during model retraining: {result.stderr}')
            st.error(f'Standard output: {result.stdout}')
    except subprocess.CalledProcessError as e:
        st.error(f'Error running train.py: {e}')
        st.error(f'Standard output: {e.stdout}')
        st.error(f'Standard error: {e.stderr}')
    except Exception as e:
        st.error(f'Exception during model retraining: {e}')

# Function to refresh Power BI dataset
def refresh_power_bi_dataset():
    try:
        # Replace with your Power BI credentials and dataset details
        tenant_id = 'your_tenant_id'
        client_id = 'your_client_id'
        client_secret = 'your_client_secret'
        dataset_id = 'your_dataset_id'

        # Get access token
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'https://analysis.windows.net/powerbi/api/.default'
        }
        token_r = requests.post(token_url, data=token_data)
        token_r.raise_for_status()
        access_token = token_r.json().get('access_token')

        # Trigger dataset refresh
        refresh_url = f"https://api.powerbi.com/v1.0/myorg/datasets/{dataset_id}/refreshes"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        refresh_data = {
            'notifyOption': 'MailOnFailure'
        }
        refresh_r = requests.post(refresh_url, headers=headers, json=refresh_data)
        refresh_r.raise_for_status()

        st.success('Power BI dataset refresh initiated successfully.')

    except requests.exceptions.RequestException as e:
        st.error(f'Error refreshing Power BI dataset: {e}')

# Streamlit app
def main():
    st.title('Car Price Prediction')
    st.write('This app predicts the price of a car based on user inputs and displays a Power BI report.')

    # Show warning message if model needs refresh
    if st.session_state.model_needs_refresh:
        st.warning('The SQL table has been updated. Please refresh the model to ensure predictions are up-to-date.')

    # Page selector in the sidebar
    page = st.sidebar.selectbox('Choose a page:', ['Prediction', 'Power BI Report', 'Upload CSV', 'Model Management'])

    if page == 'Prediction':
        # Load model, label encoders, and feature names
        model, label_encoders, feature_names = load_assets()

        # User input options
        st.header('Input Features')

        # Input fields (customize as per your dataset columns)
        manufacturer = st.selectbox('Manufacturer', label_encoders['manufacturer_name'].classes_)
        model_name = st.selectbox('Model Name', label_encoders['model_name'].classes_)
        transmission = st.selectbox('Transmission', label_encoders['transmission'].classes_)
        color = st.selectbox('Color', label_encoders['color'].classes_)
        year = st.slider('Year Produced', min_value=2000, max_value=2024)
        engine_fuel = st.selectbox('Engine Fuel', label_encoders['engine_fuel'].classes_)
        engine_has_gas = st.selectbox('Engine Has Gas', ['TRUE', 'FALSE'])
        engine_type = st.selectbox('Engine Type', label_encoders['engine_type'].classes_)
        engine_capacity = st.slider('Engine Capacity (L)', min_value=0.0, max_value=6.0, step=0.1)
        body_type = st.selectbox('Body Type', label_encoders['body_type'].classes_)
        has_warranty = st.selectbox('Has Warranty', ['TRUE', 'FALSE'])
        state = st.selectbox('State', label_encoders['state'].classes_)

        # Predict button
        if st.button('Predict'):
            user_input = {
                'manufacturer_name': manufacturer,
                'model_name': model_name,
                'transmission': transmission,
                'color': color,
                'year_produced': year,
                'engine_fuel': engine_fuel,
                'engine_has_gas': engine_has_gas,
                'engine_type': engine_type,
                'engine_capacity': engine_capacity,
                'body_type': body_type,
                'has_warranty': has_warranty,
                'state': state
            }
            predicted_price = predict_price(model, label_encoders, feature_names, user_input)
            st.success(f'Predicted Price: ${predicted_price:.2f}')

    elif page == 'Power BI Report':
        st.header('Power BI Report')
        # Replace with your actual Power BI report URL
        power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=d27042f0-8b52-4139-8fe7-1e5595fbd2fe&autoAuth=true&ctid=8c25f501-4f7e-4b41-9aa6-bdd5f317d4e4"
        st.markdown(f'<iframe width="100%" height="400" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)

    elif page == 'Upload CSV':
        st.header('Upload CSV Data')

        # File upload for CSV
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read CSV data
            try:
                df = pd.read_csv(uploaded_file)
                st.write('Preview of uploaded data:')
                st.write(df.head())

                # Connect to MySQL
                conn = connect_to_mysql()
                if conn:
                    # Insert data into MySQL
                    insert_data_to_mysql(conn, df)

                    conn.close()

            except Exception as e:
                st.error(f'Error uploading file: {e}')

    elif page == 'Model Management':
        st.header('Model Management')
        last_refresh = read_last_refresh_date()
        st.write(f'Last model refresh: {last_refresh}')
        if st.button('Refresh Model'):
            run_model_retraining()
            refresh_power_bi_dataset()

if __name__ == '__main__':
    main()
