import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Set the page configuration for the Streamlit app
st.set_page_config(page_title="Flight Price Prediction", page_icon=">>", layout="centered")

# App title and description
st.title("Flight Price Prediction App")
st.markdown("Enter your flight details below to get an estimated ticket price using Decision Tree Regressor Model")

# Cache the model and data to avoid reloading every time
@st.cache_resource
def load_model_and_data():
    # Path to the trained model
    model_path = r"C:\Users\Fluxtech\Downloads\decision_tree_flight_model.pkl"
    # Path to the flight dataset
    dataset_path = r"C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
    
    # Load the trained model
    model = joblib.load(model_path)
    
    # Load the dataset
    data = pd.read_csv(dataset_path)
    return model, data

# Load model and dataset
model, data = load_model_and_data()

# Preprocess categorical columns: trim whitespace and lowercase text
for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    le.fit(data[col])
    label_encoders[col] = le

# Subheader for user input section
st.subheader("Input Flight Details")

# Input widgets for user to select flight details
airline_input = st.selectbox("Airline", sorted(data['Airline'].unique()))
source_input = st.selectbox("Source", sorted(data['Source'].unique()))
destination_input = st.selectbox("Destination", sorted(data['Destination'].unique()))
stops_input = st.selectbox("Stops", sorted(data['Total_Stops'].unique()))
duration_hours = st.number_input("Duration (hours)", min_value=0.0, step=0.5)
month_input = st.number_input("Month", min_value=1, max_value=12, step=1)

# Function to safely encode categorical input
def safe_transform(encoder, value):
    value = value.strip().lower()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.warning(f"'{value}' not found in training data. Using fallback (-1)")
        return -1

# Mapping flight stop descriptions to numerical values
stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

# Predict flight price when button is clicked
if st.button("Predict Price"):
    # Encode categorical inputs
    airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
    source_encoded = safe_transform(label_encoders['Source'], source_input)
    destination_encoded = safe_transform(label_encoders['Destination'], destination_input)
    
    # Encode stops
    stops_encoded = stops_mapping.get(stops_input, 0)

    # Combine all features in the same order as training
    input_data = [[airline_encoded, source_encoded, destination_encoded, stops_encoded, duration_hours, month_input]]

    # Make prediction using the trained model
    predicted_price = model.predict(input_data)
    
    # Display predicted price to user
    st.success(f"Estimated Flight Ticket Price: {predicted_price[0]:,.2f}")
