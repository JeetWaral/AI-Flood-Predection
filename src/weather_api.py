import requests
import streamlit as st

def get_weather_data(city: str):
    # Get API key from secrets.toml
    api_key = st.secrets["weatherapi"]["api_key"]

    # API URL
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"

    # Make request
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "city": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "temp_c": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "condition": data["current"]["condition"]["text"],
            "rainfall": data["current"].get("precip_mm", 0.0) 
        }
        return weather_info
    else:
        return {"error": "Unable to fetch weather data"}
