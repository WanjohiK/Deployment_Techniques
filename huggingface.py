import kagglehub  # Library to download datasets from Kaggle easily
import pandas as pd  # Data manipulation library
from sklearn.model_selection import train_test_split  # For splitting dataset into train and test
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels
from sklearn.ensemble import RandomForestClassifier  # Machine learning model
import os  # For interacting with the operating system (file paths)
import gradio as gr  # For creating a web interface for the model

print("📥 Downloading dataset from KaggleHub...")
# Download the earthquake dataset from KaggleHub
path = kagglehub.dataset_download("ahmeduzaki/earthquake-alert-prediction-dataset")


# List all CSV files in the downloaded dataset directory
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

# If no CSV files are found, raise an error
if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in the downloaded dataset folder")

# Get the full path to the first CSV file
filepath = os.path.join(path, csv_files[0])
print(f"✅ Using dataset file: {filepath}")

# ✅ FIX 2: Load the dataset safely
# Load the CSV dataset into a pandas DataFrame
data = pd.read_csv(filepath)
print("✅ Dataset loaded successfully")
print("📊 Columns:", data.columns.tolist())  # Print column names for verification

# ✅ FIX 3: y should be a Series, not DataFrame
# Select feature columns for the model
X = data[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
# Select target column
y = data['alert']

# Encode labels (convert categorical alert levels into numeric values)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier with 100 trees, max depth of 8, and fixed random state
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=8
)
# Train the model using the training dataset
rf_model.fit(X_train, y_train)

# Evaluate model accuracy on the test dataset
accuracy = rf_model.score(X_test, y_test)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")  # Print accuracy in percentage

# Prediction function to use with Gradio interface
def predict_earthquake_alert(magnitude, depth, cdi, mmi, sig):
    # Create a DataFrame from user input to match model input format
    user_input = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]],
                              columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    # Predict encoded alert level
    pred_encoded = rf_model.predict(user_input)[0]
    # Convert numeric prediction back to original alert label
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    # Return a readable prediction string
    return f"Predicted Earthquake Alert Level: {pred_label}"

# Gradio interface for interactive web app
interface = gr.Interface(
    fn=predict_earthquake_alert,  # Function to call when inputs are submitted
    inputs=[  # Input fields for user to provide earthquake parameters
        gr.Number(label="Magnitude"),
        gr.Number(label="Depth"),
        gr.Number(label="CDI"),
        gr.Number(label="MMI"),
        gr.Number(label="SIG")
    ],
    outputs=gr.Textbox(label="Prediction"),  # Output field to display predicted alert
    title="🌍 Earthquake Alert Prediction",  # App title
    description="Enter earthquake parameters to predict the alert level using a Random Forest Classifier model."  # App description
)

# Launch the Gradio app when script is run directly
if __name__ == "__main__":
    interface.launch()  # Opens interactive web app in browser
