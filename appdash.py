import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = dash .Dash(__name__)
app.title = "Flight Price Ticket Prediction App"

def load_model_and_data():
    model_path = r"C:\Users\Fluxtech\Downloads\decision_tree_flight_model.pkl"
    dataset_path = r"C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
    model = joblib.load(model_path)
    data = pd.read_csv(dataset_path)
    return model, data

model, data = load_model_and_data()

for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

label_encoders = {}
for col in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    le.fit(data[col])
    label_encoders[col] = le

def safe_transform(encoder, value):
    value = value.strip().lower()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

app.layout = html.Div(
    style={'maxWidth':'600px', 'margin':'auto', 'padding':'20px', 'fontFamily':'Arial'},
    children=[
        html.H1("Flight Price Prediction App", style={'textAlign':'center'}),
        html.P("Enter Your flight details below to get estimated ticket price using Decision Tree Regressor model"),
        html.Hr(),
        html.H3("Input Flight Details"),

        html.Label('Airline'),
        dcc.Dropdown(
            id='airline-input',
            options=[{'label':i.title(), 'value':i} for i in sorted(data['Airline'].unique())],
            value = sorted(data['Airline'].unique())[0]),

        html.Label("Source"),
        dcc.Dropdown(
            id='Source-input',
            options=[{'label':i.title(), 'value':i} for i in sorted(data['Source'].unique())],
            value = sorted(data['Source'].unique())[0]),

        html.Label("Destination"),
        dcc.Dropdown(
            id='Destination-input',
            options=[{'label': i.title(), 'value': i} for i in sorted(data['Destination'].unique())],
            value=sorted(data['Destination'].unique())[0]),

        html.Label("Total Stops"),
        dcc.Input(id='stops-input',type='number', min=0, step=0.5, value=1.0),

        html.Label("Duration(hours)"),
        dcc.Input(id='duration-input',type='number', min=0, step=0.5, value=1.0),

        html.Label("Month (1-12)"),
        dcc.Input(id='month-input',type='number', min=1, max=12, step=1, value=1.0),
        html.Br(), html.Br(),
        html.Button("predict Price", id='predict-price-button', n_clicks=0),
        html.Br(),html.Br(),

        html.Div(id='prediction-output', style={'fontSize': '20px', 'fontWeight': 'bold'}),
        html.Hr()
    ]
)

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
    if n_clicks > 0:
        airline_encoded = safe_transform(label_encoders['Airline'], airline)
        source_encoded = safe_transform(label_encoders['Source'], Source)
        destination_encoded = safe_transform(label_encoders['Destination'], Destination)
        # Match the feature order
        input_data = [
            [airline_encoded, source_encoded, destination_encoded, stops, duration, month]]
        predicted_price = model.predict(input_data)[0]
        result_text = "Estimated Flight Ticket Price: {:.2f}".format(predicted_price)
        return result_text
    else:
        return ""

if __name__ == '__main__':
    app.run(debug=True)
