import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from flask import Flask, render_template_string, request

app = Flask(__name__)
file_path = "C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
data = pd.read_csv(file_path)

# Clean and lowercase categorical columns
for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Define features and target
features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_hours', 'Month']
target = "Price"

X = data.loc[:, features].copy()
y = data[target]

# Convert categorical columns using LabelEncoder
label_encoders = {}
for col in ['Airline', 'Source', 'Destination']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Convert Total_Stops to numeric if needed
# Example mapping (you can adjust based on your dataset)
stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
X['Total_Stops'] = X['Total_Stops'].map(stops_mapping).fillna(0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)

def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"'{value}' not found in training data. Using fallback (-1)")
        return -1

HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Flight Dataset</title>
<style>
    body{
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f0f2f5;}
    .container{
        background: white;
        padding: 30px 40px;
        border-radius: 10px;
        box-shadow: 0 4px 10px 0 rgba(0,0,0,0.1);
        text-align: center;
        width: 350px;
    }
    h2 {
        margin-bottom: 20px;
    }
    input[type="text"], input[type="number"] {
        width:90%;
        padding: 8px;
        margin: 80px 0;
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

    h3 {
        margin-top: 20px;
        color: white;
    }

</style>
</head>
<body>
<div class="container">
<h2>Flight Price Prediction</h2>
<form method="post">
    <input type="submit" name="Airline" placeholder="Airline"><br>
    <input type="submit" name="Source" placeholder="Source"><br>
    <input type="submit" name="Destination" placeholder="Destination"><br>
    <input type="number" step="0.1" name="Total_Stops" placeholder="Total_Stops"><br>
    <input type="number" step="0.1" name="Duration_hours" placeholder="Duration Hours"><br>
    <input type="number" name="Month" placeholder="Month (1-12)"><br>
    <input type="submit" value="Predict" placeholder="Price"><br>
</form>
{% if price %}
<h3> Estimated Price: {{price}}</h3>
{% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])

def predict():
    price = None
    if request.method == 'POST':
        airline_input = request.form["Airline"].strip().lower()
        source_input = request.form["Source"].strip().lower()
        destination_input = request.form["Destination"].strip().lower()
        stops_input = float(request.form["Total Stops (e.g. 'non-stop', '1 stop', '2 stops')"])
        duration_hours = float(request.form["Duration Hours"])
        month_input = int(request.form["Month (1-12)"])
        airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
        source_encoded = safe_transform(label_encoders['Source'], source_input)
        destination_encoded = safe_transform(label_encoders['Destination'], destination_input)
        stops_encoded = stops_mapping.get(stops_input, 0)
        input_data = [
            [airline_encoded, source_encoded, destination_encoded, stops_encoded, duration_hours, month_input]]
        predicted_price = model.predict(input_data)
        price = "{:.2f}".format(predicted_price[0])
        return render_template_string(HTML_FORM, price=price)

if __name__ == '__main__':
    app.run(debug=True)






# import os
# import sqlite3
# import pandas as pd
# from flask import Flask, render_template_string, request, redirect, url_for, session
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from werkzeug.security import generate_password_hash, check_password_hash
#
# app = Flask(__name__)
# app.secret_key = "supersecretkey"  # change this for production
#
# DB_FILE = "users.db"
#
# # ---------- DATABASE SETUP ----------
# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE NOT NULL,
#             password TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()
#
# init_db()
#
# # ---------- LOAD MODEL AND DATA ----------
# file_path = r"C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"Dataset not found at {file_path}")
#
# data = pd.read_csv(file_path)
# for col in ['Airline', 'Source', 'Destination']:
#     data[col] = data[col].astype(str).str.strip().str.lower()
#
# features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_hours', 'Month']
# target = "Price"
# X = data[features].copy()
# y = data[target]
#
# label_encoders = {}
# for col in ['Airline', 'Source', 'Destination']:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     label_encoders[col] = le
#
# stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
# X['Total_Stops'] = X['Total_Stops'].map(stops_mapping).fillna(0).astype(int)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = DecisionTreeRegressor(max_depth=6, random_state=42)
# model.fit(X_train, y_train)
#
# def safe_transform(encoder, value):
#     return encoder.transform([value])[0] if value in encoder.classes_ else -1
#
# # ---------- HTML TEMPLATES ----------
# LOGIN_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Login</title>
# <style>
#     body {font-family: Arial; background-color:#eef2f3; display:flex; justify-content:center; align-items:center; height:100vh;}
#     .box {background:white; padding:30px; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,0.1);}
#     input {display:block; width:100%; padding:8px; margin:10px 0;}
#     button {width:100%; padding:10px; background:#007bff; border:none; color:white; cursor:pointer; border-radius:5px;}
#     button:hover {background:#0056b3;}
#     a {text-decoration:none; color:#007bff;}
# </style>
# <script>
# function validateForm() {
#   let user = document.forms["loginForm"]["username"].value;
#   let pass = document.forms["loginForm"]["password"].value;
#   if (user === "" || pass === "") {
#     alert("Both fields are required!");
#     return false;
#   }
# }
# </script>
# </head>
# <body>
# <div class="box">
#   <h2>Login</h2>
#   <form name="loginForm" method="post" onsubmit="return validateForm()">
#     <input type="text" name="username" placeholder="Username" required>
#     <input type="password" name="password" placeholder="Password" required>
#     <button type="submit">Login</button>
#   </form>
#   <p>Don't have an account? <a href="{{url_for('register')}}">Register here</a></p>
#   {% if error %}<p style="color:red;">{{error}}</p>{% endif %}
# </div>
# </body>
# </html>
# """
#
# REGISTER_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Register</title>
# <style>
#     body {font-family: Arial; background-color:#eef2f3; display:flex; justify-content:center; align-items:center; height:100vh;}
#     .box {background:white; padding:30px; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,0.1);}
#     input {display:block; width:100%; padding:8px; margin:10px 0;}
#     button {width:100%; padding:10px; background:#28a745; border:none; color:white; cursor:pointer; border-radius:5px;}
#     button:hover {background:#218838;}
#     a {text-decoration:none; color:#007bff;}
# </style>
# <script>
# function validateForm() {
#   let user = document.forms["regForm"]["username"].value;
#   let pass = document.forms["regForm"]["password"].value;
#   if (user.length < 3) { alert("Username too short"); return false; }
#   if (pass.length < 5) { alert("Password too short"); return false; }
# }
# </script>
# </head>
# <body>
# <div class="box">
#   <h2>Create Account</h2>
#   <form name="regForm" method="post" onsubmit="return validateForm()">
#     <input type="text" name="username" placeholder="Username" required>
#     <input type="password" name="password" placeholder="Password" required>
#     <button type="submit">Register</button>
#   </form>
#   <p>Already have an account? <a href="{{url_for('login')}}">Login here</a></p>
#   {% if error %}<p style="color:red;">{{error}}</p>{% endif %}
# </div>
# </body>
# </html>
# """
#
# PREDICT_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <title>Flight Price Prediction</title>
# <style>
#     body {font-family: Arial; background:#f5f5f5; display:flex; justify-content:center; align-items:center; height:100vh;}
#     .container {background:white; padding:30px; border-radius:8px; box-shadow:0 4px 10px rgba(0,0,0,0.1); width:400px;}
#     input {width:95%; padding:8px; margin:8px 0;}
#     button {padding:10px 20px; background:#007bff; color:white; border:none; border-radius:5px; cursor:pointer;}
#     button:hover {background:#0056b3;}
#     a {color:#007bff;}
# </style>
# <script>
# function validatePredictionForm() {
#   let inputs = document.querySelectorAll("input[type=text], input[type=number]");
#   for (let i of inputs) {
#     if (i.value.trim() === "") {
#       alert("Please fill out all fields.");
#       return false;
#     }
#   }
# }
# </script>
# </head>
# <body>
# <div class="container">
#   <h2>Welcome, {{session['username']}}</h2>
#   <h3>Flight Price Prediction</h3>
#   <form method="post" onsubmit="return validatePredictionForm()">
#     <input type="text" name="Airline" placeholder="Airline (e.g. indigo)" required><br>
#     <input type="text" name="Source" placeholder="Source (e.g. delhi)" required><br>
#     <input type="text" name="Destination" placeholder="Destination (e.g. mumbai)" required><br>
#     <input type="text" name="Total_Stops" placeholder="Total Stops (e.g. 1 stop)" required><br>
#     <input type="number" step="0.1" name="Duration_hours" placeholder="Duration Hours" required><br>
#     <input type="number" name="Month" placeholder="Month (1-12)" required><br>
#     <button type="submit">Predict</button>
#   </form>
#   {% if price %}
#     <h3>Estimated Price: ₹{{price}}</h3>
#   {% endif %}
#   <p><a href="{{url_for('logout')}}">Logout</a></p>
# </div>
# </body>
# </html>
# """
#
# # ---------- ROUTES ----------
# @app.route('/')
# def home():
#     return redirect(url_for('login'))
#
# @app.route('/login', methods=['GET','POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#
#         conn = sqlite3.connect(DB_FILE)
#         cursor = conn.cursor()
#         cursor.execute("SELECT password FROM users WHERE username=?", (username,))
#         result = cursor.fetchone()
#         conn.close()
#
#         if result and check_password_hash(result[0], password):
#             session['username'] = username
#             return redirect(url_for('predict'))
#         else:
#             error = "Invalid username or password"
#
#     return render_template_string(LOGIN_TEMPLATE, error=error)
#
# @app.route('/register', methods=['GET','POST'])
# def register():
#     error = None
#     if request.method == 'POST':
#         username = request.form['username']
#         password = generate_password_hash(request.form['password'])
#
#         try:
#             conn = sqlite3.connect(DB_FILE)
#             cursor = conn.cursor()
#             cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#             conn.commit()
#             conn.close()
#             return redirect(url_for('login'))
#         except sqlite3.IntegrityError:
#             error = "Username already exists."
#     return render_template_string(REGISTER_TEMPLATE, error=error)
#
# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     if 'username' not in session:
#         return redirect(url_for('login'))
#
#     price = None
#     if request.method == 'POST':
#         airline_input = request.form["Airline"].strip().lower()
#         source_input = request.form["Source"].strip().lower()
#         destination_input = request.form["Destination"].strip().lower()
#         stops_input = request.form["Total_Stops"].strip().lower()
#         duration_hours = float(request.form["Duration_hours"])
#         month_input = int(request.form["Month"])
#
#         airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
#         source_encoded = safe_transform(label_encoders['Source'], source_input)
#         destination_encoded = safe_transform(label_encoders['Destination'], destination_input)
#         stops_encoded = stops_mapping.get(stops_input, 0)
#
#         input_data = [[airline_encoded, source_encoded, destination_encoded, stops_encoded, duration_hours, month_input]]
#         predicted_price = model.predict(input_data)
#         price = f"{predicted_price[0]:,.2f}"
#
#     return render_template_string(PREDICT_TEMPLATE, price=price)
#
# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login'))
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5050)













# import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from flask import Flask, render_template_string, request
# import os
#
# app = Flask(__name__)
#
# # ✅ FIX 1: Use raw string or double backslashes in Windows paths
# file_path = r"C:\Users\Fluxtech\Downloads\archive (1)\flight_dataset.csv"
#
# # ✅ Check file existence
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"Dataset not found at {file_path}")
#
# data = pd.read_csv(file_path)
#
# # ✅ Clean and lowercase categorical columns
# for col in ['Airline', 'Source', 'Destination']:
#     data[col] = data[col].astype(str).str.strip().str.lower()
#
# # ✅ Define features and target
# features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_hours', 'Month']
# target = "Price"
#
# X = data.loc[:, features].copy()
# y = data[target]
#
# # ✅ Convert categorical columns using LabelEncoder
# label_encoders = {}
# for col in ['Airline', 'Source', 'Destination']:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     label_encoders[col] = le
#
# # ✅ Map total stops to numeric
# stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
# X['Total_Stops'] = X['Total_Stops'].map(stops_mapping).fillna(0).astype(int)
#
# # ✅ Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # ✅ Train model
# model = DecisionTreeRegressor(max_depth=6, random_state=42)
# model.fit(X_train, y_train)
#
# # ✅ Safe label transform
# def safe_transform(encoder, value):
#     if value in encoder.classes_:
#         return encoder.transform([value])[0]
#     else:
#         print(f"'{value}' not found in training data. Using fallback (-1)")
#         return -1
#
# # ✅ HTML form template (fixed input fields)
# HTML_FORM = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <title>Flight Price Prediction</title>
# <style>
#     body {
#         font-family: Arial, sans-serif;
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         height: 100vh;
#         background-color: #f0f2f5;
#     }
#     .container {
#         background: white;
#         padding: 30px 40px;
#         border-radius: 10px;
#         box-shadow: 0 4px 10px rgba(0,0,0,0.1);
#         text-align: center;
#         width: 350px;
#     }
#     h2 { margin-bottom: 20px; }
#     input[type="text"], input[type="number"] {
#         width: 90%;
#         padding: 8px;
#         margin: 10px 0;
#         font-size: 16px;
#     }
#     input[type="submit"] {
#         padding: 10px 20px;
#         font-size: 16px;
#         background-color: #007bff;
#         color: white;
#         border: none;
#         border-radius: 5px;
#         cursor: pointer;
#         margin-top: 10px;
#     }
#     input[type="submit"]:hover {
#         background-color: #0056b3;
#     }
#     h3 { margin-top: 20px; color: #333; }
# </style>
# </head>
# <body>
# <div class="container">
#     <h2>Flight Price Prediction</h2>
#     <form method="post">
#         <input type="text" name="Airline" placeholder="Airline (e.g. indigo)" required><br>
#         <input type="text" name="Source" placeholder="Source (e.g. delhi)" required><br>
#         <input type="text" name="Destination" placeholder="Destination (e.g. mumbai)" required><br>
#         <input type="text" name="Total_Stops" placeholder="Total Stops (e.g. 1 stop)" required><br>
#         <input type="number" step="0.1" name="Duration_hours" placeholder="Duration Hours" required><br>
#         <input type="number" name="Month" placeholder="Month (1-12)" required><br>
#         <input type="submit" value="Predict"><br>
#     </form>
#     {% if price %}
#         <h3>Estimated Price: ₹{{price}}</h3>
#     {% endif %}
# </div>
# </body>
# </html>
# """
#
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     price = None
#     if request.method == 'POST':
#         # ✅ Extract and clean inputs
#         airline_input = request.form["Airline"].strip().lower()
#         source_input = request.form["Source"].strip().lower()
#         destination_input = request.form["Destination"].strip().lower()
#         stops_input = request.form["Total_Stops"].strip().lower()
#         duration_hours = float(request.form["Duration_hours"])
#         month_input = int(request.form["Month"])
#
#         # ✅ Encode safely
#         airline_encoded = safe_transform(label_encoders['Airline'], airline_input)
#         source_encoded = safe_transform(label_encoders['Source'], source_input)
#         destination_encoded = safe_transform(label_encoders['Destination'], destination_input)
#         stops_encoded = stops_mapping.get(stops_input, 0)
#
#         # ✅ Prepare input and predict
#         input_data = [[airline_encoded, source_encoded, destination_encoded, stops_encoded, duration_hours, month_input]]
#         predicted_price = model.predict(input_data)
#         price = f"{predicted_price[0]:,.2f}"
#
#     return render_template_string(HTML_FORM, price=price)
#
# if __name__ == '__main__':
#     app.run(debug=True)











#
#
