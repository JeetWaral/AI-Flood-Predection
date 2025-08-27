import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model_bundle = joblib.load("flood_model.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
encoders = model_bundle["encoders"]

# Page config
st.set_page_config(page_title="Flood Prediction AI", page_icon="🌊", layout="wide")

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        /* Background Gradient */
        .main {
            background: linear-gradient(135deg, #caf0f8, #90e0ef, #00b4d8, #0077b6);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 15px;
        }

        /* Titles */
        h1 {
            font-family: 'Segoe UI', sans-serif;
            font-size: 42px;
            font-weight: 800;
            color: #03045e;
            text-shadow: 1px 1px 2px #90e0ef;
        }

        h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: 600;
            color: #0077b6;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #0077b6, #00b4d8);
            color: white;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 28px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #00b4d8, #90e0ef);
            transform: scale(1.05);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }

        /* Result Card */
        .result-card {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }

        /* Table Styling */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            margin-top: 15px;
        }
        .dataframe th {
            background-color: #0077b6;
            color: white !important;
            padding: 10px;
            text-align: center;
            font-size: 16px;
        }
        .dataframe td {
            background-color: #f8f9fa;
            text-align: center;
            padding: 10px;
            font-size: 15px;
        }
        .dataframe tr:nth-child(even) td {
            background-color: #e3f2fd;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown("<h1 style='text-align:center;'>🌊 AI Flood Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter environmental conditions below to check flood risk in your region.</p>", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:15px; 
                background:linear-gradient(90deg,#00b4d8,#48cae4,#90e0ef);
                border-radius:12px; color:white; font-size:20px; font-weight:bold;">
        🌍 Flood Risk Input Panel
    </div>
    """, unsafe_allow_html=True)

st.sidebar.write("### 📌 Essential Parameters")
st.sidebar.caption("Fill these fields to get accurate flood risk predictions.")

with st.sidebar.expander("🌐 Location Details", expanded=True):
    latitude = st.text_input("🌍 Latitude", placeholder="-90 to 90", help="Enter geographic latitude (e.g., 6.00)")
    longitude = st.text_input("📍 Longitude", placeholder="-180 to 180", help="Enter geographic longitude (e.g., 77.00)")

with st.sidebar.expander("🌦️ Environmental Factors", expanded=True):
    rainfall = st.text_input("🌧️ Rainfall (mm)", placeholder="0 to 2000", help="Total rainfall in millimeters")
    elevation = st.text_input("⛰️ Elevation (m)", placeholder="-400 to 9000", help="Height above sea level in meters")

with st.sidebar.expander("🌊 Hydrological Factors", expanded=True):
    river_discharge = st.text_input("🌊 River Discharge (m³/s)", placeholder="0 to 100000", help="Flow of water in cubic meters per second")
    water_level = st.text_input("📈 Water Level (m)", placeholder="0 to 100", help="Measured water level in meters")

# 🎛️ Advanced (Hidden by default, shows defaults)
with st.sidebar.expander("🔧 Advanced Parameters (Defaults Applied)", expanded=False):
    st.info("These parameters are set to defaults but can be adjusted if needed.")
    st.write(f"🌡️ Temperature: **{25.0} °C**")
    st.write(f"💧 Humidity: **{70}%**")
    st.write(f"🟩 Land Cover: **2**")
    st.write(f"🌱 Soil Type: **1**")
    st.write(f"👥 Population Density: **500 per km²**")
    st.write(f"🏗️ Infrastructure: **2**")
    st.write(f"📜 Historical Floods: **1**")

# 🎨 Sidebar footer
st.sidebar.markdown(
    """
    <div style="margin-top:20px; padding:12px; text-align:center;
                background:linear-gradient(90deg,#90e0ef,#48cae4,#00b4d8);
                border-radius:10px; color:white; font-size:14px;">
        🌟 Tip: Adjust key values above and click Predict to see results!
    </div>
    """, unsafe_allow_html=True)


# ----------------- Safe Conversion -----------------
def safe_cast(val, to_type, default=0):
    try:
        return to_type(val)
    except:
        return default

latitude = safe_cast(latitude, float)
longitude = safe_cast(longitude, float)
rainfall = safe_cast(rainfall, float)
river_discharge = safe_cast(river_discharge, float)
water_level = safe_cast(water_level, float)
elevation = safe_cast(elevation, float)

# Default values for less important inputs
temperature = 25.0
humidity = 70.0
land_cover = 2
soil_type = 1
population_density = 500
infrastructure = 2
historical_floods = 1

# ----------------- DataFrame -----------------
input_df = pd.DataFrame([[
    latitude, longitude, rainfall, temperature, humidity,
    river_discharge, water_level, elevation, land_cover,
    soil_type, population_density, infrastructure,
    historical_floods
]], columns=[
    "Latitude", "Longitude", "Rainfall", "Temperature", "Humidity",
    "River Discharge", "Water Level", "Elevation", "Land Cover",
    "Soil Type", "Population Density", "Infrastructure",
    "Historical Floods"
])

X_scaled = scaler.transform(input_df)

# ----------------- Layout -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Data Preview")

    # Show exactly what user entered in sidebar
    preview_df = input_df[[
        "Latitude", "Longitude", "Rainfall",
        "River Discharge", "Water Level", "Elevation"
    ]].round(2).T.reset_index()

    preview_df.columns = ["Parameter", "Value"]
    preview_df["Value"] = preview_df["Value"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    st.table(preview_df)


with col2:
    st.subheader("🧮 Model Prediction")
    if st.button("🚀 Predict Flood Risk"):
        prediction = model.predict(X_scaled)[0]
        if prediction == 1:
            st.markdown("<div class='result-card' style='background:#ffcccc; color:#d90429;'>⚠️ High Risk: Flood Likely!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card' style='background:#ccffcc; color:#006400;'>✅ Safe: Flood Not Likely</div>", unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color:#222;'>🌍 Developed with ❤️ by <b>Team FloodGuard AI</b></p>", unsafe_allow_html=True)
