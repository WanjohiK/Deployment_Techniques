import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


file_path = r"C:\Users\Fluxtech\Downloads\archive\earthquake_alert_balanced_dataset.csv"
data = pd.read_csv(file_path)
X = data[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
y = data[['alert']]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=8
)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

app = FastAPI(title="Earth Alert Predictor API")
@app.get("/")

def root():
    return {"message": "Earth Alert Predictor API. Use POST / predict with JSON input"}

class EarthquakeInput(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float

@app.post("/predict")

def predict(input_data: EarthquakeInput):
    df = pd.DataFrame([[
        input_data.magnitude,
        input_data.depth,
        input_data.cdi,
        input_data.mmi,
        input_data.sig
    ]], columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    pred_encoded = rf_model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return {"predicted_alert": pred_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

