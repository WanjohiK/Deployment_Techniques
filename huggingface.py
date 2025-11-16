import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import gradio as gr

# Download the dataset from KaggleHub
print("📥 Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("ahmeduzaki/earthquake-alert-prediction-dataset")

# List CSV files in the downloaded dataset folder
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

# Raise an error if no CSV file is found
if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in the downloaded dataset folder")

# Construct the full path to the CSV file
filepath = os.path.join(path, csv_files[0])
print(f"✅ Using dataset file: {filepath}")

# Load the dataset into a pandas DataFrame
data = pd.read_csv(filepath)
print("✅ Dataset loaded successfully")
print("📊 Columns:", data.columns.tolist())

# Select features and target
X = data[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
y = data['alert']

# Encode target labels into numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=8
)
rf_model.fit(X_train, y_train)

# Evaluate and display model accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Define a prediction function for Gradio interface
def predict_earthquake_alert(magnitude, depth, cdi, mmi, sig):
    # Prepare input as DataFrame
    user_input = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]],
                              columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    # Predict encoded class
    pred_encoded = rf_model.predict(user_input)[0]
    # Convert encoded class back to original label
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return f"Predicted Earthquake Alert Level: {pred_label}"

# Build Gradio interface
interface = gr.Interface(
    fn=predict_earthquake_alert,
    inputs=[
        gr.Number(label="Magnitude"),
        gr.Number(label="Depth"),
        gr.Number(label="CDI"),
        gr.Number(label="MMI"),
        gr.Number(label="SIG")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🌍 Earthquake Alert Prediction",
    description="Enter earthquake parameters to predict the alert level using a Random Forest Classifier model."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()
