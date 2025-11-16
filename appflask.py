"""
===========================================================
 FLIGHT PRICE PREDICTION - FLASK DEPLOYMENT 
===========================================================

This script trains a Decision Tree Regressor on a flight 
price dataset and deploys it using a simple Flask web app.

Users can input flight details (Airline, Source, Destination,
Stops, Duration, Month) and receive a predicted ticket price.
===========================================================
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from flask import Flask, render_template_string, request

# Initialize Flask app
app = Flask(__name__)

# ===========================================================
# 1) LOAD DATASET
# ===========================================================
file_path = "C:\\Users\\Fluxtech\\Downloads\\archive (1)\\flight_dataset.csv"
data = pd.read_csv(file_path)

# Clean and standardize categorical text columns
for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

# ===========================================================
# 2) FEATURE SELECTION
# ===========================================================
features = ['Airline', 'Source', 'Destination', 'Total_Stops', 
            'Duration_hours', 'Month']
target = "Price"

X = data.loc[:, features].copy()
y = data[target]

# ===========================================================
# 3) ENCODE CATEGORICAL COLUMNS
# ===========================================================
label_encoders = {}

for col in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoder for later use in prediction

# Mapping for total stops (string → numeric)
stops_mapping = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}

# Convert stops column using mapping
X['Total_Stops'] = X['Total_Stops'].map(stops_mapping).fillna(0).astype(int)

# ===========================================================
# 4) SPLIT DATA AND TRAIN MODEL
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)

# ===========================================================
# 5) SAFE TRANSFORM FOR UNKNOWN CATEGORIES
# ===========================================================
def safe_transform(encoder, value):
    """
    Converts a category to its encoded numeric value.
    If the category was NOT seen during training, return -1.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"'{value}' not found in training data. Using fallback (-1)")
        return -1

# ===========================================================
# 6) HTML FORM (INLINE TEMPLATE)
# ===========================================================
HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Flight Dataset</title>
<style>
    body {
        font-family: Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background-color: #f0f2f5;
    }
    .container {
        background: white;
        padding: 30px 40px;
        border-radius: 10px;
        box-shadow: 0 4px 10px 0 rgba(0,0,0,0.1);
        text-align: center;
        width: 350px;
    }
    h2 { margin-bottom: 20px; }
    input[type="text"], input[type="number"] {
        width: 90%;
        padding: 8px;
        margin: 8px 0;
        font-size: 16px;
    }
    input[type="submit"] {
        padding: 10px 20px;
        font-size: 16px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 10px;
    }
    input[type="submit"]:hover {
        background-color: #0056b3;
    }
    h3 { margin-top: 20px; }
</style>
</head>
<body>
<div class="container">
<h2>Flight Price Prediction</h2>

<form method="post">
    <input type="text" name="Airline" placeholder="Airline"><br>
    <input type="text" name="Source" placeholder="Source"><br>
    <input type="text" name="Destination" placeholder="Destination"><br>
    <input type="text" name="Total Stops (e.g. 'non-stop', '1 stop')" placeholder="Total Stops"><br>
    <input type="number" step="0.1" name="Duration Hours" placeholder="Duration Hours"><br>
    <input type="number" name="Month (1-12)" placeholder="Month (1-12)"><br>

    <input type="submit" value="Predict"><br>
</form>

{% if price %}
<h3> Estimated Price: {{price}}</h3>
{% endif %}

</div>
</body>
</html>
"""

# ===========================================================
# 7) FLASK ROUTE — PREDICTION
# ===========================================================
@app.route('/', methods=['GET', 'POST'])
def predict():
    price = None

    if request.method == 'POST':

        # Read inputs from HTML form
        airline_input = request.form["Airline"].strip().lower()
        source_input = request.form["Source"].strip().lower()
        destination_input = request.form["Destination"].strip().lower()

        # Stops entered as string (e.g. "1 stop")
        stops_input = request.form["Total Stops (e.g. 'non-stop', '1 stop')"]

        duration_hours = float(request.form["Duration Hours"])
        month_input = int(request.form["Month (1-12)"])

        # Encode user inputs
        airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
        source_encoded = safe_transform(label_encoders['Source'], source_input)
        destination_encoded = safe_transform(label_encoders['Destination'], destination_input)

        # Map stops
        stops_encoded = stops_mapping.get(stops_input, 0)

        # Prepare data for prediction
        input_data = [[
            airline_encoded, 
            source_encoded, 
            destination_encoded, 
            stops_encoded, 
            duration_hours, 
            month_input
        ]]

        # Predict price
        predicted_price = model.predict(input_data)
        price = "{:.2f}".format(predicted_price[0])

        return render_template_string(HTML_FORM, price=price)

    return render_template_string(HTML_FORM, price=None)

# ===========================================================
# 8) RUN APPLICATION
# ===========================================================
if __name__ == '__main__':
    app.run(debug=True)
