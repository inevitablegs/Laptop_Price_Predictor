
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open('pipe1.pkl', 'rb') as file:
    rf = pickle.load(file)

# Confirm the model type
if not hasattr(rf, "predict"):
    st.error("The loaded object is not a model. Please check the 'laptop_data.pkl' file.")


# Load the data for UI element options
data = pd.read_csv("traineddata.csv")

st.title("Laptop Price Predictor")

# User inputs
company = st.selectbox('Brand', data['Company'].unique())
type = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('OS', data['OpSys'].unique())
weight = st.number_input('Weight of the laptop', min_value=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size', min_value=1.0)
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['CPU_name'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', data['Gpu brand'].unique())

# Convert categorical inputs
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Calculate PPI
X_resolution, Y_resolution = map(int, resolution.split('x'))
ppi = ((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5 / screen_size

# Construct the input array
query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
query = query.reshape(1, -1)

if st.button('Predict Price'):
    try:
        # Prepare the input array as usual
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, -1)

        # Ensure the model can handle unseen categories
        prediction = int(np.exp(rf.predict(query)[0]))

        st.title(f"Predicted price for this laptop could be between {prediction-1000}₹ to {prediction+1000}₹")
    except ValueError as e:
        st.error(f"An error occurred during prediction: {e}")

