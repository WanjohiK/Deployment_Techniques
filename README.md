# Machine Learning Projects - Multi-Framework Deployment

This repository contains **two machine learning projects** deployed using **multiple frameworks** to demonstrate different deployment approaches for ML models.

## 📋 Projects Overview

### Project 1: Flight Price Prediction
Predicts flight ticket prices based on flight parameters using a Decision Tree Regressor model.

**Input Features:**
- Airline (e.g., Indigo, Air India)
- Source city (e.g., Delhi, Mumbai)
- Destination city (e.g., Bangalore, Kolkata)
- Total stops (non-stop, 1 stop, 2 stops, etc.)
- Duration in hours
- Month of travel (1-12)

**Output:** Estimated ticket price

**Model:** Decision Tree Regressor
**Model File:** `decision_tree_flight_model.pkl`
**Dataset:** `flight_dataset.csv`

**Deployment Methods:**
- ✅ **Dash** (`appdash.py`)
- ✅ **Flask** (`appflask.py`)
- ✅ **Streamlit** (`streamlitapp.py`)

---

### Project 2: Earthquake Alert Prediction
Predicts earthquake alert levels based on seismic parameters using a Random Forest Classifier.

**Input Features:**
- Magnitude
- Depth
- CDI (Community Decimal Intensity)
- MMI (Modified Mercalli Intensity)
- SIG (Significance)

**Output:** Alert level (green, yellow, orange, red)

**Model:** Random Forest Classifier (100 estimators, max_depth=8)
**Dataset:** `earthquake_alert_balanced_dataset.csv` (auto-downloaded via KaggleHub)

**Deployment Methods:**
- ✅ **FastAPI** (`modelfastapi.py`)
- ✅ **Gradio** (`huggingface.py`)

---

## 🚀 Getting Started

### Prerequisites

Install required packages based on which deployment method you want to use:

```bash
# For Flight Price Prediction
pip install pandas scikit-learn joblib

# For Dash deployment
pip install dash

# For Flask deployment
pip install flask

# For Streamlit deployment
pip install streamlit

# For Earthquake Alert Prediction
pip install pandas scikit-learn

# For FastAPI deployment
pip install fastapi uvicorn

# For Gradio deployment
pip install gradio kagglehub
```

**Or install everything at once:**
```bash
pip install dash flask streamlit fastapi uvicorn gradio pandas scikit-learn joblib kagglehub
```

### Installation

1. **Clone this repository:**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Install dependencies** (see Prerequisites above)

3. **Prepare required files:**

**For Flight Price Prediction:**
- Model file: `decision_tree_flight_model.pkl`
- Dataset: `flight_dataset.csv`
- Update file paths in scripts (`appdash.py`, `appflask.py`, `streamlitapp.py`)

**For Earthquake Alert Prediction:**
- Dataset will be auto-downloaded via KaggleHub (for Gradio)
- For FastAPI, download `earthquake_alert_balanced_dataset.csv` manually

---

## 📱 Deployment Methods

## Project 1: Flight Price Prediction

### Option 1: Dash (appdash.py)
**Best for:** Interactive dashboards with real-time updates

**Run:**
```bash
python appdash.py
```
**Access:** `http://127.0.0.1:8050`

**Features:**
- Modern, reactive UI with callbacks
- Dropdown menus for categorical inputs
- Real-time prediction updates
- Professional dashboard layout

---

### Option 2: Flask (appflask.py)
**Best for:** Traditional web applications with custom HTML/CSS

**Run:**
```bash
python appflask.py
```
**Access:** `http://127.0.0.1:5000`

**Features:**
- Simple HTML form interface
- Custom styling with embedded CSS
- Lightweight and flexible
- Session management code included (commented out)

**Note:** Flask version includes commented authentication system (login/register) that you can enable.

---

### Option 3: Streamlit (streamlitapp.py)
**Best for:** Rapid prototyping and quick ML demos

**Run:**
```bash
streamlit run streamlitapp.py
```

**Features:**
- Clean, intuitive UI
- Automatic caching (`@st.cache_resource`) for better performance
- Built-in widgets (selectbox, number_input)
- Hot reloading during development
- No HTML/CSS required

---

## Project 2: Earthquake Alert Prediction

### Option 4: FastAPI (modelfastapi.py)
**Best for:** Building production REST APIs

**Run:**
```bash
python modelfastapi.py
```
**Access:** `http://127.0.0.1:8000`
**API Docs:** `http://127.0.0.1:8000/docs` (auto-generated)

**Features:**
- RESTful API endpoints
- Automatic OpenAPI/Swagger documentation
- Pydantic data validation
- High performance with async support
- Model trains on startup

**Example API Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "magnitude": 5.5,
    "depth": 10.0,
    "cdi": 4.5,
    "mmi": 5.0,
    "sig": 500.0
  }'
```

**Response:**
```json
{
  "predicted_alert": "yellow"
}
```

---

### Option 5: Gradio (huggingface.py)
**Best for:** Quick sharing and HuggingFace Spaces deployment

**Run:**
```bash
python huggingface.py
```

**Features:**
- Shareable public/private links
- HuggingFace Spaces integration ready
- Simple interface with number inputs
- Auto-downloads dataset via KaggleHub
- Creates shareable URL automatically

**Note:** Gradio creates a local URL and a shareable public URL (expires in 72 hours).

---

## 🗂️ Project Structure

```
.
├── appdash.py                          # Flight - Dash deployment
├── appflask.py                         # Flight - Flask deployment  
├── streamlitapp.py                     # Flight - Streamlit deployment
├── modelfastapi.py                     # Earthquake - FastAPI deployment
├── huggingface.py                      # Earthquake - Gradio deployment
├── decision_tree_flight_model.pkl      # Pre-trained flight model
├── flight_dataset.csv                  # Flight training dataset
├── earthquake_alert_balanced_dataset.csv # Earthquake dataset (optional)
└── README.md                           # This file
```

---

## 🔧 Configuration

### Update File Paths

All scripts use absolute paths. Update these to match your system:

**Flight Price Prediction Apps:**
```python
# Windows
model_path = r"C:\Users\YourName\Downloads\decision_tree_flight_model.pkl"
dataset_path = r"C:\Users\YourName\Downloads\flight_dataset.csv"

# Mac/Linux
model_path = "/Users/YourName/Downloads/decision_tree_flight_model.pkl"
dataset_path = "/Users/YourName/Downloads/flight_dataset.csv"
```

**Earthquake Alert (FastAPI):**
```python
# Windows
file_path = r"C:\Users\YourName\Downloads\earthquake_alert_balanced_dataset.csv"

# Mac/Linux
file_path = "/Users/YourName/Downloads/earthquake_alert_balanced_dataset.csv"
```

**Earthquake Alert (Gradio):**
No path configuration needed - dataset auto-downloads via KaggleHub!

---

## 📊 Model Details

### Project 1: Flight Price Prediction

**Algorithm:** Decision Tree Regressor

**Features (6 total):**
1. Airline (categorical - label encoded)
2. Source (categorical - label encoded)
3. Destination (categorical - label encoded)
4. Total_Stops (numeric: 0-4)
5. Duration_hours (numeric)
6. Month (numeric: 1-12)

**Preprocessing:**
```python
# Categorical cleaning
for col in ['Airline', 'Source', 'Destination']:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Stops mapping
stops_mapping = {
    'non-stop': 0, 
    '1 stop': 1, 
    '2 stops': 2, 
    '3 stops': 3, 
    '4 stops': 4
}
```

---

### Project 2: Earthquake Alert Prediction

**Algorithm:** Random Forest Classifier

**Hyperparameters:**
- n_estimators: 100
- max_depth: 8
- random_state: 42

**Features (5 total):**
1. magnitude (float)
2. depth (float)
3. cdi (float) - Community Decimal Intensity
4. mmi (float) - Modified Mercalli Intensity
5. sig (float) - Significance

**Output Classes:** Alert levels (green, yellow, orange, red)

**Accuracy:** Typically 90%+ on test set

---

## 🎯 Framework Comparison

| Feature | Dash | Flask | Streamlit | FastAPI | Gradio |
|---------|------|-------|-----------|---------|--------|
| **Project** | Flight | Flight | Flight | Earthquake | Earthquake |
| **Ease of Use** | Medium | Medium | Very Easy | Medium | Very Easy |
| **Setup Time** | Medium | Low | Very Low | Medium | Very Low |
| **UI Quality** | Excellent | Good | Excellent | N/A (API) | Good |
| **Customization** | High | Very High | Medium | N/A | Low |
| **API Support** | No | Manual | No | Native | No |
| **Documentation** | Manual | Manual | Auto | Auto (Swagger) | Auto |
| **Best For** | Dashboards | Full Web Apps | ML Demos | Production APIs | Quick Sharing |
| **Learning Curve** | Moderate | Low | Very Low | Moderate | Very Low |

---

## 🚀 When to Use Each Framework

### For Flight Price Prediction:

**Choose Dash when:**
- Building interactive dashboards with multiple components
- Need real-time updates and callbacks
- Want a professional UI without writing HTML/CSS
- Building internal company dashboards

**Choose Flask when:**
- You want complete control over HTML/CSS/JavaScript
- Need to integrate with existing web infrastructure
- Want to add authentication/sessions (code included)
- Building a full-featured web application

**Choose Streamlit when:**
- Rapidly prototyping your ML model
- Want the fastest path from model to app
- Sharing with data science teams
- Don't want to write any frontend code
- Need quick demos for stakeholders

---

### For Earthquake Alert Prediction:

**Choose FastAPI when:**
- Building production REST APIs
- Need to integrate with mobile apps or other services
- Want automatic API documentation (Swagger UI)
- Require high performance and async support
- Building microservices architecture

**Choose Gradio when:**
- Sharing models publicly with non-technical users
- Deploying to HuggingFace Spaces
- Need shareable links quickly (public URLs)
- Building simple demo interfaces
- Want zero configuration deployment

---

## 🐛 Troubleshooting

### Common Issues

**1. File Not Found Error**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution:** Update file paths to match your system
```python
# Windows - use raw strings with 'r'
model_path = r"C:\Your\Path\model.pkl"

# Mac/Linux - use forward slashes
model_path = "/your/path/model.pkl"
```

**2. Module Not Found**
```
ModuleNotFoundError: No module named 'dash'
```
**Solution:** Install the missing package
```bash
pip install dash  # or flask, streamlit, fastapi, gradio
```

**3. Model Loading Error**
```
ValueError: sklearn version mismatch
```
**Solution:** Match scikit-learn version
```bash
pip install scikit-learn==1.3.0  # adjust version as needed
```

**4. Port Already in Use**
```
OSError: [Errno 48] Address already in use
```
**Solution:** Change port or kill existing process
```bash
# Mac/Linux
lsof -ti:8050 | xargs kill -9

# Windows
netstat -ano | findstr :8050
taskkill /PID <PID> /F
```

**5. Unknown Input Values (Flight App)**
```
Warning: 'xyz airline' not found in training data
```
**Solution:** Check `flight_dataset.csv` for valid values or use the fallback encoding

**6. KaggleHub Download Issues (Earthquake Gradio)**
```
Error downloading dataset
```
**Solution:** Ensure you have internet connection and KaggleHub is properly installed
```bash
pip install --upgrade kagglehub
```

---

## ⚡ Quick Start Guide

### For Complete Beginners

**Want to predict Flight Prices?**

1. Install Streamlit (easiest option):
```bash
pip install streamlit pandas scikit-learn joblib
```

2. Update paths in `streamlitapp.py`

3. Run:
```bash
streamlit run streamlitapp.py
```

---

**Want to predict Earthquake Alerts?**

1. Install Gradio (easiest option):
```bash
pip install gradio pandas scikit-learn kagglehub
```

2. Run:
```bash
python huggingface.py
```

3. Dataset downloads automatically!

---

### Framework Selection Flowchart

```
START
│
├─ Building a Web App?
│  ├─ Yes → Need full control?
│  │  ├─ Yes → Use Flask
│  │  └─ No → Want interactive dashboard?
│  │     ├─ Yes → Use Dash
│  │     └─ No → Use Streamlit
│  │
│  └─ No → Building an API?
│     └─ Yes → Use FastAPI
│
└─ Just want to share quickly?
   └─ Use Gradio
```

---

## 📝 Additional Notes

### Flight Price Project
- **Flask version** includes commented authentication system (login/register with SQLite)
- All deployments handle unknown categorical values gracefully with fallback encoding
- **Streamlit** uses `@st.cache_resource` for optimized model loading
- Model is pre-trained and loaded from pickle file

### Earthquake Alert Project
- **FastAPI** trains model on startup (takes a few seconds)
- **Gradio** auto-downloads dataset from KaggleHub
- Both achieve ~90%+ accuracy on test data
- Model uses Random Forest with 100 trees

---

## 🎓 Learning Resources

### Framework Documentation
- [Dash Documentation](https://dash.plotly.com/) - Interactive dashboards
- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework
- [Streamlit Documentation](https://docs.streamlit.io/) - ML app builder
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Modern API framework
- [Gradio Documentation](https://gradio.app/docs/) - ML interfaces

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Decision Trees Guide](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)

### Deployment Guides
- [Deploying Streamlit Apps](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [Dash Enterprise](https://dash.plotly.com/deployment)

---

## 🤝 Contributing

Contributions are welcome! You can:
- Add new deployment methods
- Improve existing implementations
- Add more ML projects
- Enhance documentation
- Submit bug fixes

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

Fluxtech

---

## 🎯 Project Summary

This repository demonstrates:

✅ **Two different ML projects** solving different problems
✅ **Five deployment frameworks** with different strengths
✅ **Real-world deployment patterns** for ML models
✅ **Framework comparison** to help you choose the right tool

### Projects:
1. **Flight Price Prediction** → Dash, Flask, Streamlit
2. **Earthquake Alert Prediction** → FastAPI, Gradio

### Key Takeaway:
Different deployment methods suit different needs. Choose based on your use case, technical requirements, and target audience.

---

Made with ❤️ using Python & Machine Learning