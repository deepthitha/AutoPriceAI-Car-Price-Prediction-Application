# Car Price Prediction Application

This repository contains a car price prediction application built with Streamlit and a machine learning model. The application predicts the price of a car based on various input features.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Front End](#front-end)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/car-price-prediction.git
    cd car-price-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have MySQL installed and running. Set up your database and update the database connection details in `app.py`.

## Usage

### Model Training

1. To train the machine learning model, run:
    ```bash
    python train.py
    ```

2. The trained model will be saved as `.joblib` files.

### Front End

1. To start the Streamlit application, run:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501` to access the application.

## Project Structure

```
car-price-prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── models/
    └── model.joblib
```

- `app.py`: The Streamlit front end application.
- `train.py`: The script for training the machine learning model.
- `requirements.txt`: The dependencies required to run the project.
- `models/`: Directory where the trained model is saved.

## Dataset

The dataset contains various attributes of car listings such as manufacturer, model, year produced, engine type, and price. The dataset is stored in a MySQL database.

### Columns

- `manufacturer_name`
- `model_name`
- `transmission`
- `color`
- `odometer_value`
- `year_produced`
- `engine_fuel`
- `engine_has_gas`
- `engine_type`
- `engine_capacity`
- `body_type`
- `has_warranty`
- `state`
- `drivetrain`
- `price_usd`
- `is_exchangeable`
- `location_region`
- `number_of_photos`
- `up_counter`
- `feature_0`
- `feature_1`
- `feature_2`
- `feature_3`
- `feature_4`
- `feature_5`
- `feature_6`
- `feature_7`
- `feature_8`
- `feature_9`
- `duration_listed`

## Model Training

The machine learning model is trained using the XGBoost algorithm. The `train.py` script loads the dataset from the MySQL database, preprocesses the data, trains the model, and saves the trained model.

## Front End

The front end is built using Streamlit. It provides an interactive interface for users to input car details and get the predicted price. The application connects to the MySQL database to fetch the data and uses the trained model for predictions.

## Technologies Used

- Python
- Streamlit
- XGBoost
- MySQL

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
