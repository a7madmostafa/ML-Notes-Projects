# 🚗 Used Cars Price Prediction

This project builds a **Machine Learning pipeline** to predict **used car prices**.  
It covers the full workflow: data exploration, cleaning, preprocessing, model training, evaluation, and deployment via a Streamlit app.

---

## 📂 Project Structure

### Apps
- `apps/app.py` → Basic script for loading and testing the trained model.  
- `apps/streamlitApp.py` → Streamlit web app for interactive car price predictions.  

### Data
- `data/train-data.csv` → Raw dataset.  
- `data/cleaned_data.csv` → Preprocessed dataset ready for training.  
- `data/unprocessed_data.pkl` → Raw pickle version of dataset.  
- `data/preprocessed_data.pkl` → Preprocessed pickle version.  

### Models
- `models/ml_model.pkl` → Trained machine learning model (serialized with pickle).  

### Notebooks
1. **01- Data Exploration.ipynb** → Initial EDA (distributions, outliers, correlations).  
2. **02- Data Cleaning & Feature Engineering.ipynb** → Cleaning, handling missing values, feature creation.  
3. **03- Data Preprocessing.ipynb** → Encoding categorical variables, scaling, imputation.  
4. **04- Modelling.ipynb** → Training baseline and advanced ML models.  
5. **05- Cross Validation & Hyperparameter Tuning.ipynb** → Model evaluation and tuning.  

### Reports
- `reports/first-exploration.html` → Exported HTML report with EDA and findings.  

