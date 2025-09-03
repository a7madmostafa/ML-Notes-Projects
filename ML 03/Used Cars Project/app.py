import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('ml_model.pkl', 'rb'))
print("Model loaded successfully.")

# Example car data for prediction
car_data = pd.DataFrame({
    'Location': ['Pune'],
    'Year': [2009],
    'Kilometers_Driven': [85000],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Manual'],
    'Owner_Type': ['Third'],
    'Mileage': [30.59],
    'Engine': [998],
    'Power': [66.10],
    'Brand': ['Maruti'],
    'Model': ['A-Star']
})


# Make predictions
prediction = loaded_model.predict(car_data)
price = np.expm1(prediction) 

# Print the predicted price
print("Predicted Price:", prediction[0])