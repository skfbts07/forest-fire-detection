import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pydeck as pdk
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# Custom threshold for fire detection
FIRE_THRESHOLD = 0.3

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌲 Forest Fire Detection App")

st.sidebar.header("Input Satellite Data")

# Manual coordinate entry with direction labels
latitude_value = st.sidebar.number_input("Latitude", value=19.2000, format="%.6f")
latitude_dir = st.sidebar.selectbox("Latitude Direction", ["N", "S"])
longitude_value = st.sidebar.number_input("Longitude", value=81.7000, format="%.6f")
longitude_dir = st.sidebar.selectbox("Longitude Direction", ["E", "W"])

# Convert to signed coordinates
latitude = latitude_value if latitude_dir == "N" else -latitude_value
longitude = longitude_value if longitude_dir == "E" else -longitude_value

brightness = st.sidebar.number_input("Brightness", value=340.0, format="%.6f")
scan = st.sidebar.number_input("Scan", value=1.3, format="%.6f")
track = st.sidebar.number_input("Track", value=1.2, format="%.6f")
acq_time = st.sidebar.number_input("Acq Time", value=1145.0, format="%.6f")
confidence = st.sidebar.number_input("Confidence", value=90.0, format="%.6f")
bright_t31 = st.sidebar.number_input("Brightness T31", value=300.0, format="%.6f")
frp = st.sidebar.number_input("FRP", value=32.0, format="%.6f")

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Fire"):
    input_data = pd.DataFrame([[
        latitude, longitude, brightness, scan, track,
        confidence, bright_t31, frp
    ]], columns=['latitude','longitude','brightness','scan','track',
                 'confidence','bright_t31','frp'])

    proba = model.predict_proba(input_data)[0]
    fire_prob = proba[1]
    no_fire_prob = proba[0]
    fire_detected = fire_prob > FIRE_THRESHOLD

    st.subheader("Prediction Result")
    if fire_detected:
        st.error("🔥 Fire Detected")
    else:
        st.success("✅ No Fire Detected")

    st.write(f"Confidence (No Fire): {no_fire_prob:.3f}")
    st.write(f"Confidence (Fire): {fire_prob:.3f}")
    st.write(f"Coordinates: {latitude_value:.6f}° {latitude_dir}, {longitude_value:.6f}° {longitude_dir}")

    # Reverse geocoding
    geolocator = Nominatim(user_agent="fire_app")
    try:
        location = geolocator.reverse((latitude, longitude), language="en")
        st.write(f"📍 Location: {location.address}")
    except:
        st.write("📍 Location: Not found")

    # Map visualization
    st.pydeck_chart(pdk.Deck(
        map_style="dark",
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=6,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame({"lat":[latitude],"lon":[longitude]}),
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]' if fire_detected else '[0, 200, 30, 160]',
                get_radius=50000,
            ),
        ],
    ))

    # Line chart of fire probability over time (example using Acq Time)
    st.subheader("Fire Probability Trend")
    times = [acq_time-20, acq_time-10, acq_time, acq_time+10, acq_time+20]
    probs = [fire_prob*0.5, fire_prob*0.7, fire_prob, fire_prob*1.1, fire_prob*0.9]

    fig, ax = plt.subplots()
    ax.plot(times, probs, marker="o", color="red")
    ax.set_xlabel("Acquisition Time")
    ax.set_ylabel("Fire Probability")
    ax.set_title("Fire Probability vs Time")
    st.pyplot(fig)