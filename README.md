# FloodSenseAI
End-to-end machine learning project for predicting flood risk in Pakistan using Random Forest, Open-Meteo API, and Streamlit.
# 🌊 FloodSense AI

> **AI-Powered Flood Prediction System for Pakistan**

FloodSense AI is a Machine Learning-based flood prediction system that estimates flood risk using historical weather data and geographical features. The project helps provide early flood warnings by predicting the probability of flooding for selected districts in Pakistan.

---

## 🚀 Project Overview

FloodSense AI combines weather information and machine learning to estimate flood risk.

The application uses:

- 🌧 Daily Precipitation
- 🌦 Rainfall (Last 7 Days)
- 🌡 Mean Temperature
- 🌡 Temperature Change
- 📍 District
- 🌍 Latitude & Longitude
- 🏔 Elevation

The trained Random Forest model predicts flood probability and classifies the risk level into:

- 🟢 Low Risk
- 🟡 Moderate Risk
- 🟠 High Risk
- 🔴 Very High Risk

---

## ✨ Features

- 🌊 Flood Risk Prediction
- 🤖 Machine Learning Model
- 📊 Flood Probability Score
- ⚠ Risk Classification
- 📍 Pakistan District Selection
- 🌧 Weather-Based Prediction
- 🎯 Threshold Optimized Prediction
- 💻 Interactive Streamlit Dashboard

---

## 🧠 Machine Learning Workflow

```text
Historical Weather Data
          │
          ▼
Data Cleaning
          │
          ▼
Feature Engineering
          │
          ▼
Random Forest Model
          │
          ▼
Probability Prediction
          │
          ▼
Flood Risk Classification
```

---

## 📊 Model Information

**Algorithm**

- Random Forest Classifier

**Performance**

| Metric | Score |
|---------|-------|
| Accuracy | 91.65% |
| ROC-AUC | 0.817 |
| Threshold | 0.25 |

The prediction threshold was optimized from **0.50** to **0.25** to improve flood detection performance.

---

## 📁 Project Structure

```text
FloodSenseAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── FloodSenseAI_Best_Model.pkl
├── district_encoder.pkl
├── feature_columns.pkl
├── model_config.pkl
│
└── screenshots/
```

---

## 📂 Dataset

The model was trained using weather observations and flood-event information.

### Weather Features

- Temperature Mean
- Daily Precipitation
- Rainfall (Last 7 Days)
- Temperature Change

### Geographic Features

- District
- Latitude
- Longitude
- Elevation

### Engineered Features

- Heavy Rain Indicator

---

## 📦 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Joblib
- Open-Meteo API

---

## 📈 Model Comparison

Three machine learning algorithms were evaluated.

| Model | ROC-AUC |
|--------|---------|
| ✅ Random Forest | **0.817** |
| XGBoost | 0.779 |
| Gradient Boosting | 0.778 |

Random Forest was selected as the final deployment model.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/FloodSenseAI.git
```

Go into the project folder

```bash
cd FloodSenseAI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🖥 Application Preview

The application provides:

- Home Dashboard
- Flood Prediction Page
- Model Information
- About Project

Users enter weather information and receive:

- Flood Probability
- Risk Level
- Safety Recommendations

---

## 🌍 Data Sources

- Open-Meteo Historical Weather API
- Historical Flood Event Records
- Geographic District Information

---

## 🔮 Future Improvements

- Live Weather API Integration
- Interactive Pakistan Flood Map
- River Water Level Data
- Soil Moisture Integration
- Satellite Data
- NDMA Live Alerts
- Deep Learning Models
- Mobile-Friendly Dashboard

---

## 👨‍💻 Developer

**Shujaat Ali Khan**

Machine Learning & AI Enthusiast

Pakistan 🇵🇰

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is released under the MIT License.

Feel free to use, modify, and improve it for educational and research purposes.
