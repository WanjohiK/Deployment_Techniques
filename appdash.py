# -------------------------------------------------------------
# Import necessary libraries:
# - Dash for building the web app UI
# - pandas for handling dataset operations
# - joblib for loading the trained ML model
# - sklearn preprocessing for encoding categorical variables
# -------------------------------------------------------------
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------
# Initialize the Dash application and set the browser tab title
# -------------------------------------------------------------
app = dash .Dash(__name__)
app.title = "Flight Price Ticket Prediction App"

# -------------------------------------------------------------
# Function to load the pre-trained Decision Tree model
# and the dataset used during training.
# Ensures that the model and encoders match the training data.
# -------------------------------------------------------------
def load_model_and_data():
    model_path = r"C:\Users\Fluxtech\Downloads\decision_tree_flight_model.pkl"
    dataset_path = r"C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)
    return model, data

# -------------------------------------------------------------
# Load the ML model and dataset into memory
# -------------------------------------------------------------
model, data = load_model_and_data()

# -------------------------------------------------------------
# Standardize categorical text fields by converting them to
# lowercase and stripping unnecessary whitespace, ensuring
# encoding consistency with the training phase.
# -------------------------------------------------------------
for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

# -------------------------------------------------------------
# Create LabelEncoders for each categorical column.
# Each encoder is fitted using the original training dataset.
# Encoders are stored for use during user input transformations.
# -------------------------------------------------------------
label_encoders = {}
for col in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    le.fit(data[col])
    label_encoders[col] = le

# -------------------------------------------------------------
# Function to safely transform user input values using the same
# LabelEncoder used during training. If a new/unseen category
# appears, return a fallback value (-1) to prevent errors.
# -------------------------------------------------------------
def safe_transform(encoder, value):
    value = value.strip().lower()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

# -------------------------------------------------------------
# Define the layout of the Dash web application.
# This section defines the UI including:
# - Dropdowns for categorical features
# - Numeric inputs for numerical features
# - A button to trigger prediction
# - A div to display the predicted ticket price
# -------------------------------------------------------------
app.layout = html.Div(
    style={'maxWidth':'600px', 'margin':'auto', 'padding':'20px', 'fontFamily':'Arial'},
    children=[
        html.H1("Flight Price Prediction App", style={'textAlign':'center'}),
        html.P("Enter Your flight details below to get estimated ticket price using Decision Tree Regressor model"),
        html.Hr(),
        html.H3("Input Flight Details"),

        # Airline selection dropdown
        html.Label('Airline'),
        dcc.Dropdown(
            id='airline-input',
            options=[{'label':i.title(), 'value':i} for i in sorted(data['Airline'].unique())],
            value = sorted(data['Airline'].unique())[0]),

        # Source city dropdown
        html.Label("Source"),
        dcc.Dropdown(
            id='Source-input',
            options=[{'label':i.title(), 'value':i} for i in sorted(data['Source'].unique())],
            value = sorted(data['Source'].unique())[0]),

        # Destination city dropdown
        html.Label("Destination"),
        dcc.Dropdown(
            id='Destination-input',
            options=[{'label': i.title(), 'value': i} for i in sorted(data['Destination'].unique())],
            value=sorted(data['Destination'].unique())[0]),

        # Total stops numeric input
        html.Label("Total Stops"),
        dcc.Input(id='stops-input',type='number', min=0, step=0.5, value=1.0),

        # Duration numeric input
        html.Label("Duration(hours)"),
        dcc.Input(id='duration-input',type='number', min=0, step=0.5, value=1.0),

        # Travel month numeric input
        html.Label("Month (1-12)"),
        dcc.Input(id='month-input',type='number', min=1, max=12, step=1, value=1.0),
        html.Br(), html.Br(),

        # Prediction button
        html.Button("predict Price", id='predict-price-button', n_clicks=0),
        html.Br(),html.Br(),

        # Output display for predicted price
        html.Div(id='prediction-output', style={'fontSize': '20px', 'fontWeight': 'bold'}),
        html.Hr()
    ]
)

# -------------------------------------------------------------
# Callback function:
# Runs when user clicks the "Predict Price" button.
# - Retrieves user input
# - Encodes categorical features
# - Prepares model input in correct order
# - Generates flight price prediction
# - Displays price on the interface
# -------------------------------------------------------------
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-price-button', 'n_clicks'),
    State('airline-input', 'value'),
    State('Source-input', 'value'),
    State('Destination-input', 'value'),
    State('stops-input', 'value'),
    State('duration-input', 'value'),
    State('month-input', 'value'),
)

def predict_price(n_clicks, airline, Source, Destination, stops, duration, month):
    # Only run prediction after the user clicks the button
    if n_clicks > 0:

        # Encode inputs safely using previously fitted encoders
        airline_encoded = safe_transform(label_encoders['Airline'], airline)
        source_encoded = safe_transform(label_encoders['Source'], Source)
        destination_encoded = safe_transform(label_encoders['Destination'], Destination)

        # Maintain correct feature order for the model
        input_data = [
            [airline_encoded, source_encoded, destination_encoded, stops, duration, month]]

        # Generate prediction using the Decision Tree model
        predicted_price = model.predict(input_data)[0]

        # Format output text
        result_text = "Estimated Flight Ticket Price: {:.2f}".format(predicted_price)
        return result_text
    else:
        return ""

# -------------------------------------------------------------
# Run the Dash server in debug mode
# -------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
