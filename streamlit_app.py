# ==========================================================
# FloodSense AI
# Pakistan Flood Prediction System
# Version 1.0
# Developer: Shujaat Ali Khan
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="FloodSense AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#f5f9ff;
}

h1{
    color:#005792;
}

h2{
    color:#005792;
}

.stButton>button{
    width:100%;
    background:#005792;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0077b6;
    color:white;
}

.metric-card{

    background:white;

    padding:20px;

    border-radius:12px;

    box-shadow:0px 2px 10px rgba(0,0,0,.10);

    margin-bottom:15px;

}

.footer{

    text-align:center;

    color:gray;

    font-size:14px;

    margin-top:40px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD FILES
# ==========================================================

MODEL_PATH = "FloodSenseAI_Best_Model.pkl"
ENCODER_PATH = "district_encoder.pkl"
FEATURE_PATH = "feature_columns.pkl"
CONFIG_PATH = "model_config.pkl"

@st.cache_resource
def load_resources():

    model = joblib.load(MODEL_PATH)

    district_encoder = joblib.load(ENCODER_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    config = joblib.load(CONFIG_PATH)

    return model,district_encoder,feature_columns,config


try:

    model,district_encoder,feature_columns,config = load_resources()

except Exception as e:

    st.error("Unable to load model files.")

    st.exception(e)

    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/floods.png",
    width=90
)

st.sidebar.title("🌊 FloodSense AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "🌊 Predict Flood",

        "📊 Model Information",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Random Forest Model")

st.sidebar.info(
"""
Threshold : 0.25

Algorithm :

Random Forest
"""
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def risk_level(probability):

    if probability < 0.25:
        return "🟢 LOW"

    elif probability < 0.50:
        return "🟡 MODERATE"

    elif probability < 0.75:
        return "🟠 HIGH"

    else:
        return "🔴 VERY HIGH"


def recommendation(probability):

    if probability < 0.25:

        return """
✅ Flood risk is low.

• Continue monitoring weather.

• No immediate action required.
"""

    elif probability < 0.50:

        return """
⚠ Moderate Flood Risk

• Stay informed.

• Monitor rainfall updates.

• Prepare emergency contacts.
"""

    elif probability < 0.75:

        return """
⚠ High Flood Risk

• Avoid river crossings.

• Prepare emergency supplies.

• Follow local authority updates.
"""

    else:

        return """
🚨 VERY HIGH FLOOD RISK

• Evacuate if instructed.

• Move valuables to safe locations.

• Stay away from rivers.

• Follow NDMA instructions.
"""

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🌊 FloodSense AI")

    st.subheader("Pakistan Flood Prediction System")

    st.markdown("---")

    col1,col2 = st.columns([2,1])

    with col1:

        st.markdown("""
### Welcome!

FloodSense AI is a Machine Learning based Flood Prediction
System developed for Pakistan.

The application predicts flood risk using

- 🌧 Rainfall
- 🌡 Temperature
- 📍 District
- 🏔 Elevation
- 📈 Rainfall (Last 7 Days)
- 🌦 Temperature Change

The prediction model has been trained using historical
weather and flood event data.

Use the navigation panel on the left to start predicting
flood risk.
""")

    with col2:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

        st.metric(
            "ROC-AUC",
            "0.817"
        )

        st.metric(
            "Threshold",
            "0.25"
        )

    st.markdown("---")

    st.markdown("## Features")

    c1,c2,c3 = st.columns(3)

    with c1:

        st.success("🌧 Weather Based Prediction")

    with c2:

        st.success("📊 Machine Learning Model")

    with c3:

        st.success("⚠ Flood Risk Classification")

    st.markdown("---")

    st.info(
"""
### Project Workflow

Weather Data

⬇

Feature Engineering

⬇

Random Forest Model

⬇

Flood Probability

⬇

Risk Classification
"""
    )

    st.markdown(
        "<div class='footer'>FloodSense AI Version 1.0</div>",
        unsafe_allow_html=True
    )
# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🌊 Predict Flood":

    st.title("🌊 Flood Prediction")

    st.markdown(
        "Enter the weather information below to predict the probability of flooding."
    )

    st.markdown("---")

    # ---------------------------------------------
    # District Selection
    # ---------------------------------------------

    districts = sorted(list(district_encoder.classes_))

    district = st.selectbox(
        "📍 Select District",
        districts
    )

    # ---------------------------------------------
    # Weather Inputs
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "🌡 Temperature Mean (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        precipitation = st.number_input(
            "🌧 Daily Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=0.1
        )

        rainfall_7days = st.number_input(
            "🌦 Rainfall (Last 7 Days)",
            min_value=0.0,
            max_value=1000.0,
            value=40.0,
            step=0.1
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            value=35.85,
            step=0.0001,
            format="%.4f"
        )

        longitude = st.number_input(
            "Longitude",
            value=71.78,
            step=0.0001,
            format="%.4f"
        )

        elevation = st.number_input(
            "Elevation (m)",
            min_value=0,
            max_value=9000,
            value=1500
        )

    temp_change = st.number_input(

        "🌡 Temperature Change",

        value=0.0,

        step=0.1

    )

    # ---------------------------------------------
    # Heavy Rain Feature
    # ---------------------------------------------

    heavy_rain = 1 if precipitation >= 25 else 0

    st.info(f"Heavy Rain Feature = {heavy_rain}")

    st.markdown("---")

    # ---------------------------------------------
    # Predict Button
    # ---------------------------------------------

    if st.button("🚀 Predict Flood Risk"):

        with st.spinner("Running AI Prediction..."):

            try:

                district_encoded = district_encoder.transform(
                    [district]
                )[0]

                input_df = pd.DataFrame({

                    "district":[district_encoded],

                    "temperature_mean":[temperature],

                    "precipitation":[precipitation],

                    "latitude":[latitude],

                    "longitude":[longitude],

                    "elevation":[elevation],

                    "heavy_rain":[heavy_rain],

                    "rainfall_7days":[rainfall_7days],

                    "temp_change":[temp_change]

                })

                # Ensure correct feature order

                input_df = input_df[feature_columns]

                probability = model.predict_proba(
                    input_df
                )[0][1]

                threshold = config["threshold"]

                prediction = int(
                    probability >= threshold
                )

            except Exception as e:

                st.error("Prediction Failed")

                st.exception(e)

                st.stop()

        st.markdown("---")

        st.header("Prediction Results")

        st.progress(float(probability))

        st.metric(

            "Flood Probability",

            f"{probability*100:.2f}%"

        )

        risk = risk_level(probability)

        st.subheader(risk)

        if prediction == 1:

            st.error("⚠ Flood Risk Detected")

        else:

            st.success("✅ No Significant Flood Risk")

        st.markdown("---")

        st.subheader("Safety Recommendation")

        st.info(
            recommendation(probability)
        )

        st.markdown("---")

        with st.expander("View Model Input"):

            st.dataframe(input_df)

        with st.expander("Prediction Details"):

            st.write("Prediction:", prediction)

            st.write("Probability:", probability)

            st.write("Threshold:", threshold)
  # ==========================================================
# MODEL INFORMATION PAGE
# ==========================================================

elif page == "📊 Model Information":

    st.title("📊 Model Performance")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Algorithm", "Random Forest")

    with col2:
        st.metric("Accuracy", "91.65 %")

    with col3:
        st.metric("ROC-AUC", "0.817")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Precision", "55.56 %")

    with col2:
        st.metric("Recall", "57.89 %")

    with col3:
        st.metric("Threshold", "0.25")

    st.markdown("---")

    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame({

        "Feature":[

            "Rainfall 7 Days",
            "Temperature Mean",
            "Temperature Change",
            "Precipitation",
            "Longitude",
            "Latitude",
            "Elevation",
            "District",
            "Heavy Rain"

        ],

        "Importance":[

            0.232,
            0.209,
            0.142,
            0.123,
            0.108,
            0.071,
            0.058,
            0.057,
            0.000

        ]

    })

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.markdown("---")

    st.subheader("Model Summary")

    st.success("""
Random Forest was selected as the final model because it
produced the highest ROC-AUC score after comparing multiple
machine learning algorithms.

Algorithms Compared

• Random Forest

• Gradient Boosting

• XGBoost

Random Forest achieved the best overall performance after
threshold optimization.
""")

    st.markdown("---")

    st.subheader("Input Features")

    st.dataframe(pd.DataFrame({

        "Feature":[

            "District",

            "Temperature Mean",

            "Precipitation",

            "Latitude",

            "Longitude",

            "Elevation",

            "Heavy Rain",

            "Rainfall 7 Days",

            "Temperature Change"

        ]

    }))

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.title("ℹ About FloodSense AI")

    st.markdown("---")

    st.header("Project")

    st.write("""
FloodSense AI is a Machine Learning based Flood Prediction
System developed to estimate flood probability using weather
and geographical information.

The objective is to support early flood awareness by
combining historical weather observations with flood-event
records.
""")

    st.markdown("---")

    st.header("Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.success("""
🐍 Python

• Pandas

• NumPy

• Joblib
""")

    with tech2:

        st.success("""
🤖 Machine Learning

• Random Forest

• Scikit-Learn

• Label Encoder
""")

    with tech3:

        st.success("""
🌍 Data Sources

• Open-Meteo API

• Historical Weather

• NDMA Flood Events
""")

    st.markdown("---")

    st.header("Workflow")

    st.info("""

Weather API

⬇

Feature Engineering

⬇

Machine Learning

⬇

Random Forest

⬇

Flood Probability

⬇

Risk Classification

""")

    st.markdown("---")

    st.header("Developer")

    st.write("""

Developer:

**Shujaat Ali Khan**

Project:

**FloodSense AI**

Version:

**1.0**

Country:

**Pakistan 🇵🇰**

""")

    st.markdown("---")

    st.success("""
Thank you for using FloodSense AI.

Stay informed.

Stay prepared.

Stay safe.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
font-size:14px;
color:gray;'>

🌊 FloodSense AI

Machine Learning Flood Prediction System

Developed using Streamlit & Scikit-Learn

© 2026

</div>
""",
unsafe_allow_html=True
)
