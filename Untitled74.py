#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load Data from Excel File
def load_data():
    file_path = "C:/Users/agus.kurniawan/Downloads/DATA PREDIKSI NK LAB 2025.xlsx"
    df = pd.read_excel(file_path)
    return df

data = load_data()

# Encoding Suppliers to numeric values
supplier_mapping = {supplier: idx for idx, supplier in enumerate(data['Suppliers'].unique())}
data['Suppliers'] = data['Suppliers'].map(supplier_mapping)

# Splitting data for training
X = data[['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING']]
y = data['GCV (ARB) LAB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = float("-inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# Streamlit UI
st.title("Prediksi GCV (ARB) LAB")

supplier_input = st.selectbox("Pilih Supplier", list(supplier_mapping.keys()))
gcv_input = st.number_input("GCV ARB UNLOADING", min_value=3000, max_value=5000, value=4200)
tm_input = st.number_input("TM ARB UNLOADING", min_value=20.0, max_value=50.0, value=35.0)
ash_input = st.number_input("Ash Content ARB UNLOADING", min_value=1.0, max_value=10.0, value=5.0)
ts_input = st.number_input("Total Sulphur ARB UNLOADING", min_value=0.0, max_value=1.0, value=0.3)

if st.button("Prediksi"):
    supplier_encoded = supplier_mapping[supplier_input]
    input_data = np.array([[supplier_encoded, gcv_input, tm_input, ash_input, ts_input]])
    prediction = best_model.predict(input_data)[0]
    st.success(f"Hasil Prediksi GCV (ARB) LAB: {prediction:.2f}")
    st.write(f"Model Terbaik: {type(best_model).__name__} (R²: {best_r2:.2f})")


# In[ ]:




