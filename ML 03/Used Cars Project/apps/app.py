import numpy as np
import pandas as pd
import os, pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ml_model.pkl")

with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully.")

# Example car data for prediction
car_data = pd.DataFrame(
    {
 'Location': 'Pune',
 'Kilometers_Driven': 115000,
 'Fuel_Type': 'Diesel',
 'Transmission': 'Manual',
 'Owner_Type': 'Second',
 'Mileage': 20.54,
 'Engine': 1598.0,
 'Power': 103.6,
 'Seats': 5.0,
 'Brand': 'Volkswagen',
 'Model': 'Vento',
 'Age': 8
}, index=[0])


# Make predictions
prediction = loaded_model.predict(car_data)
price = np.expm1(prediction) 

# Print the predicted price
print("Predicted Price:", prediction[0])