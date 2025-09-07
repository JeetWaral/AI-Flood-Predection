import streamlit as st
import pandas as pd
import joblib
from weather_api import get_weather_data

# ----------------- Load model -----------------
model_bundle = joblib.load("flood_model.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
encoders = model_bundle["encoders"]

# ----------------- Page config -----------------
st.set_page_config(page_title="Flood Prediction AI", layout="wide")

# ----------------- Initialize session state -----------------
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'city' not in st.session_state:
    st.session_state.city = None

# ----------------- CSS -----------------
st.markdown("""
    <style>
    /* Background GIF with proper isolation */
    .stApp {
        background: url("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTR3c3p5Z2lvNmdqNXlma2N6dWZ6aTRyMzA1cDRoamprNmpmcXpoZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/rW6BgNjFd8GLFeoSDd/giphy.gif") 
                    no-repeat center center fixed;
        background-size: cover;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    /* Dark overlay for contrast */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.35);
        z-index: 0;
    }

    /* Ensure app content sits above overlay */
    .block-container, section[data-testid="stSidebar"] {
        position: relative;
        z-index: 1;
    }

    /* Frosted-glass main content */
    .block-container {
        background: rgba(255,255,255,0.12);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        flex: 1;
    }
    
    /* Enhanced sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.10) !important;
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        margin: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar content container */
    .sidebar-content {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }

    /* Headings and text */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, .stCaption {
        color: #ffffff !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
        background: none !important;
    }

    /* Expander headers - FIXED VISIBILITY */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0,119,182,0.7), rgba(0,180,216,0.7)) !important;
        color: white !important;
        border-radius: 8px;
        padding: 12px 15px;
        font-weight: 600;
        margin-top: 10px;
        border: none !important;
    }
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(0,119,182,0.8), rgba(0,180,216,0.8)) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#0077b6,#00b4d8);
        color: #fff;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 22px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#00b4d8,#90e0ef);
        transform: scale(1.05);
    }

    /* Radio buttons */
    .stRadio > div {
        flex-direction: column;
        gap: 10px;
    }
    .stRadio > div > label {
        background-color: rgba(255,255,255,0.1);
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        border-left: 4px solid #4a90e2;
        color: white;
    }
    .stRadio > div > label:hover {
        background-color: rgba(255,255,255,0.2);
        border-left: 4px solid #00b4d8;
        transform: translateX(5px);
    }
    .stRadio > div > label[data-testid="stRadioLabel"] > div:first-child {
        color: white !important;
        font-weight: 600;
        font-size: 16px;
    }

    /* Result cards */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        margin: 15px 0;
    }
    .result-card.safe {
        background: rgba(200,255,200,0.85);
        color: #024b20;
    }
    .result-card.risk {
        background: rgba(255,200,200,0.85);
        color: #6b0000;
    }

    /* Tables */
    .stTable, table {
        background: rgba(255,255,255,0.15) !important;
        color: #fff !important;
        border-radius: 12px;
        padding: 12px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stTable th {
        color: #ffdd00 !important;
        font-weight: bold;
        text-align: center;
    }
    .stTable td {
        text-align: center;
    }

    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: rgba(255,255,255,0.9) !important;
        color: #000 !important;
        border-radius: 6px;
        padding: 6px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #4a90e2;
    }

    /* Hide Streamlit default header & footer */
    header[data-testid="stHeader"] {display: none !important;}
    footer {display: none !important;}

    /* Footer */
    .footer {
        width: 100%;
        text-align: center;
        font-size: 16px;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 12px;
        border-radius: 12px 12px 0 0;
        color: #fff;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.3);
        margin-top: auto;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #00b4d8;
    }
    
    /* Custom cards */
    .custom-card {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-left: 4px solid #00b4d8;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0,119,182,0.8), rgba(0,180,216,0.8));
                color: white;
                padding: 25px 20px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <h1 style="font-size: 38px; font-weight: 800; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            AI Flood Prediction System
        </h1>
        <p style="font-size: 18px; font-weight: 400; margin-top: 8px;">
            Advanced flood risk assessment using environmental and hydrological data
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:15px; 
                    background:linear-gradient(90deg,#0077b6,#00b4d8);
                    border-radius:12px; color:white; font-size:20px; font-weight:bold;">
            Flood Risk Input Panel
        </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio("Select Data Input Method:", ("Manual Input", "Fetch Live Weather"))

# ----------------- Safe Conversion -----------------
def safe_cast(val, to_type, default=0):
    try: return to_type(val)
    except: return default

# ----------------- Default Hardcoded Values -----------------
latitude, longitude, rainfall, temperature, humidity = 0, 0, 10, 25.0, 70.0
river_discharge, water_level, elevation = 1000, 5, 100
land_cover, soil_type, population_density, infrastructure, historical_floods = 2, 1, 500, 2, 1

# ----------------- Input Handling -----------------
if mode == "Manual Input":
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-content">
                <h3 style="margin-top: 0; color: white;">Manual Input Panel</h3>
                <p style="color: rgba(255,255,255,0.8); font-size: 14px;">
                    Enter precise environmental and hydrological data for accurate flood prediction
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Location Details", expanded=True):
            latitude = safe_cast(st.text_input("Latitude", placeholder="-90 to 90", help="Enter latitude coordinate between -90 and 90"), float)
            longitude = safe_cast(st.text_input("Longitude", placeholder="-180 to 180", help="Enter longitude coordinate between -180 and 180"), float)

        with st.expander("Environmental Factors", expanded=True):
            rainfall = safe_cast(st.text_input("Rainfall (mm)", placeholder="0 to 2000", help="Total rainfall in millimeters"), float)
            elevation = safe_cast(st.text_input("Elevation (m)", placeholder="-400 to 9000", help="Elevation above sea level in meters"), float)

        with st.expander("Hydrological Factors", expanded=True):
            river_discharge = safe_cast(st.text_input("River Discharge (m³/s)", placeholder="0 to 100000", help="Volume of water flowing through river per second"), float)
            water_level = safe_cast(st.text_input("Water Level (m)", placeholder="0 to 100", help="Current water level measurement"), float)

elif mode == "Fetch Live Weather":
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-content">
                <h3 style="margin-top: 0; color: white;">Live Weather Data</h3>
                <p style="color: rgba(255,255,255,0.8); font-size: 14px;">
                    Fetch real-time weather data for selected Indian cities
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        indian_cities = [
            "Mumbai", "Delhi", "Kolkata", "Chennai", "Bengaluru", 
            "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
            "Patna", "Bhopal", "Guwahati", "Thiruvananthapuram", "Kochi",
            "Varanasi", "Nagpur", "Surat", "Ranchi", "Chandigarh"
        ]

        city = st.selectbox("Select a City (India)", indian_cities, help="Choose a city to fetch live weather data")

        if st.button("Fetch Weather Data"):
            try:
                weather = get_weather_data(city)
                st.session_state.weather_data = weather
                st.session_state.city = city
                st.success(f"Weather data successfully fetched for {city}")
                
                # Override defaults with live data
                temperature = weather.get("temp_c", 25.0)
                humidity = weather.get("humidity", 70.0)
                rainfall = weather.get("rainfall", 0.0)

            except Exception as e:
                st.error(f"Error fetching weather data: {str(e)}")
        
        # Display weather metrics if data exists
        if st.session_state.weather_data:
            weather = st.session_state.weather_data
            st.markdown("---")
            st.markdown("**Current Weather Conditions**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Temperature", f"{weather['temp_c']} °C", help="Current temperature in Celsius")
                st.metric("Humidity", f"{weather['humidity']} %", help="Relative humidity percentage")
            with col2:
                st.metric("Condition", weather['condition'], help="Current weather condition")
                st.metric("Rainfall", f"{weather.get('rainfall', 0.0)} mm", help="Precipitation in millimeters")

# ----------------- DataFrame -----------------
# Use session state data if available
if st.session_state.weather_data and mode == "Fetch Live Weather":
    weather = st.session_state.weather_data
    temperature = weather.get("temp_c", 25.0)
    humidity = weather.get("humidity", 70.0)
    rainfall = weather.get("rainfall", 0.0)

input_df = pd.DataFrame([[latitude, longitude, rainfall, temperature, humidity,
    river_discharge, water_level, elevation, land_cover, soil_type,
    population_density, infrastructure, historical_floods]],
    columns=["Latitude","Longitude","Rainfall","Temperature","Humidity",
    "River Discharge","Water Level","Elevation","Land Cover","Soil Type",
    "Population Density","Infrastructure","Historical Floods"])

X_scaled = scaler.transform(input_df)

# ----------------- Layout -----------------
col1, col2 = st.columns(2)

with col1:
    if mode == "Manual Input":
        st.markdown("""
            <div class="custom-card">
                <h3>Manual Input Data</h3>
                <p>Review the parameters used for flood prediction analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        preview_df = input_df[["Latitude","Longitude","Rainfall","Temperature","Humidity",
                              "River Discharge","Water Level","Elevation"]].round(2).T
        preview_df.columns = ["Values"]
        preview_df.index.name = "Parameter"
        st.dataframe(preview_df, use_container_width=True)
        
    elif mode == "Fetch Live Weather" and st.session_state.weather_data:
        st.markdown("""
            <div class="custom-card">
                <h3>Live Weather Data</h3>
                <p>Real-time meteorological information for flood risk assessment</p>
            </div>
        """, unsafe_allow_html=True)
        
        weather = st.session_state.weather_data
        api_df = pd.DataFrame({
            "Parameter": ["City", "Temperature (°C)", "Humidity (%)", "Condition", "Rainfall (mm)"],
            "Values": [st.session_state.city, weather["temp_c"], weather["humidity"], 
                      weather["condition"], weather.get("rainfall", 0.0)]
        }).set_index("Parameter")
        
        st.dataframe(api_df, use_container_width=True)

with col2:
    st.markdown("""
        <div class="custom-card">
            <h3>Model Prediction</h3>
            <p>AI-powered flood risk assessment based on input parameters</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Analyze Flood Risk", type="primary"):
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        if prediction == 1:
            risk_level = "HIGH RISK"
            risk_color = "risk"
            risk_percentage = probability[1] * 100
            st.markdown(f"<div class='result-card {risk_color}'>{risk_level}: Flood Likely! ({risk_percentage:.1f}% probability)</div>", unsafe_allow_html=True)
            st.progress(risk_percentage/100)
            
            # Additional risk information
            st.markdown("""
                <div style="background: rgba(255,200,200,0.2); padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <h4>⚠️ Risk Advisory</h4>
                    <p>Based on current conditions, there is a high probability of flooding. Recommended actions:</p>
                    <ul>
                        <li>Monitor water levels continuously</li>
                        <li>Prepare evacuation plans if in flood-prone area</li>
                        <li>Stay updated with official weather alerts</li>
                        <li>Secure important documents and belongings</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            risk_level = "SAFE"
            risk_color = "safe"
            safe_percentage = probability[0] * 100
            st.markdown(f"<div class='result-card {risk_color}'>{risk_level}: Flood Not Likely ({safe_percentage:.1f}% confidence)</div>", unsafe_allow_html=True)
            st.progress(safe_percentage/100)
            
            # Additional safety information
            st.markdown("""
                <div style="background: rgba(200,255,200,0.2); padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <h4>✅ Safety Status</h4>
                    <p>Current conditions indicate low flood risk. Maintain regular monitoring:</p>
                    <ul>
                        <li>Continue standard weather awareness</li>
                        <li>Review emergency preparedness plans</li>
                        <li>Monitor local water body levels</li>
                        <li>Stay informed about changing conditions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Show model confidence
        st.markdown(f"""
            <div style="text-align: center; margin-top: 15px;">
                <p style="color: rgba(255,255,255,0.8);">Model Confidence: {max(probability)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("""
    <div class="footer">
        Developed by <b>Team FloodGuard AI</b> | Advanced Flood Prediction System v2.1
    </div>
""", unsafe_allow_html=True)