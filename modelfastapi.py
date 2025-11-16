import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the dataset from CSV file
file_path = r"C:\Users\Fluxtech\Downloads\archive\earthquake_alert_balanced_dataset.csv"
data = pd.read_csv(file_path)

# Select features (X) and target (y)
X = data[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
y = data[['alert']]

# Encode target labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Initialize and train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,  # number of trees in the forest
    random_state=42,   # ensures reproducibility
    max_depth=8        # maximum depth of each tree
)
rf_model.fit(X_train, y_train)

# Evaluate model accuracy on the test set
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Create a FastAPI app
app = FastAPI(title="Earth Alert Predictor API")

# Root endpoint to provide basic API information
@app.get("/")
def root():
    return {"message": "Earth Alert Predictor API. Use POST /predict with JSON input"}

# Define input data model for POST requests using Pydantic
class EarthquakeInput(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float

# Prediction endpoint: accepts JSON input and returns predicted alert level
@app.post("/predict")
def predict(input_data: EarthquakeInput):
    # Convert input JSON to a DataFrame for model prediction
    df = pd.DataFrame([[
        input_data.magnitude,
        input_data.depth,
        input_data.cdi,
        input_data.mmi,
        input_data.sig
    ]], columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    
    # Predict encoded class
    pred_encoded = rf_model.predict(df)[0]
    
    # Convert encoded prediction back to original alert label
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    
    # Return prediction as JSON
    return {"predicted_alert": pred_label}

# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
