import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Import XGBoost
from sklearn.preprocessing import LabelEncoder
import pickle

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
            print('Connected to MySQL database')
            return conn
        else:
            print('Connection to MySQL failed')
            return None
    except mysql.connector.Error as e:
        print(f'Error connecting to MySQL: {e}')
        return None

# Function to fetch data from MySQL database
def fetch_data_from_mysql(conn):
    try:
        cursor = conn.cursor()
        select_query = "SELECT * FROM car_listings"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        # Convert data to pandas DataFrame
        df = pd.DataFrame(rows, columns=['manufacturer_name', 'model_name', 'transmission', 'color',
                                         'odometer_value', 'year_produced', 'engine_fuel', 'engine_has_gas',
                                         'engine_type', 'engine_capacity', 'body_type', 'has_warranty',
                                         'state', 'drivetrain', 'price_usd', 'is_exchangeable', 'location_region',
                                         'number_of_photos', 'up_counter', 'feature_0', 'feature_1', 'feature_2',
                                         'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',
                                         'feature_9', 'duration_listed'])
        return df
    except mysql.connector.Error as e:
        print(f'Error fetching data: {e}')
        return None

# Function to preprocess data (encode categorical variables)
def preprocess_data(df):
    # Convert categorical variables into numerical format
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

# Main function for training the model
def main():
    # Connect to MySQL
    conn = connect_to_mysql()
    if conn:
        # Fetch data from MySQL
        data = fetch_data_from_mysql(conn)
        conn.close()
        
        # Perform data preprocessing and modeling
        if data is not None:
            # Preprocess data (encode categorical variables)
            data_processed, label_encoders = preprocess_data(data)
            
            # Example preprocessing steps (modify as per your requirements)
            X = data_processed.drop(['price_usd'], axis=1)  # Features
            y = data_processed['price_usd']  # Target variable

            # Example: Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train an XGBoost model
            model = XGBRegressor()
            model.fit(X_train, y_train)

            # Example: Save the trained model to a file
            with open('xgboost_model.pkl', 'wb') as f:
                pickle.dump(model, f)

            # Save label encoders to file (for future use in prediction)
            with open('label_encoders.pkl', 'wb') as f:
                pickle.dump(label_encoders, f)

            # Save feature names to file
            feature_names = X.columns.tolist()
            with open('feature_names.pkl', 'wb') as f:
                pickle.dump(feature_names, f)

            print('XGBoost model training completed and saved.')

if __name__ == '__main__':
    main()
