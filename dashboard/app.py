import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime
import subprocess
import sys
import atexit

# ==============================
# PROCESS MANAGEMENT
# ==============================
if "backend_proc" not in st.session_state:
    st.session_state.backend_proc = None
if "ai_proc" not in st.session_state:
    st.session_state.ai_proc = None
if "history" not in st.session_state:
    st.session_state.history = []

def start_system():
    # Start FastAPI Backend
    if st.session_state.backend_proc is None:
        st.session_state.backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"]
        )
    
    # Start AI Monitor (Camera)
    if st.session_state.ai_proc is None:
        st.session_state.ai_proc = subprocess.Popen(
            [sys.executable, "-m", "app.main"]
        )

def stop_system():
    # Terminate Backend
    if st.session_state.backend_proc is not None:
        st.session_state.backend_proc.terminate()
        st.session_state.backend_proc = None
        
    # Terminate AI Monitor
    if st.session_state.ai_proc is not None:
        st.session_state.ai_proc.terminate()
        st.session_state.ai_proc = None

# Ensure background processes are killed if Streamlit is stopped
atexit.register(stop_system)

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="ImpactForge AIoT Dashboard", layout="wide")

st.title("🚦 ImpactForge Crowd Intelligence System")
st.subheader("Real-time Analytics Backend Feed")

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("⚙️ System Controls")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start", type="primary", use_container_width=True):
        start_system()
with col2:
    if st.button("⏹️ Stop", use_container_width=True):
        stop_system()
        st.session_state.history = [] # Clear history on stop

status_indicator = st.sidebar.empty()

system_running = st.session_state.backend_proc is not None or st.session_state.ai_proc is not None

# ==============================
# MAIN DASHBOARD LOGIC
# ==============================
if not system_running:
    status_indicator.warning("System is offline.")
    st.info("👈 Press 'Start' in the sidebar to launch the FastAPI backend and AI Camera.")
else:
    status_indicator.success("System Processes Running")
    
    data = None
    try:
        # Fetch data from the background FastAPI server
        response = requests.get("http://127.0.0.1:8000/status", timeout=1)
        response.raise_for_status()
        fetched_data = response.json()
        
        # Check if the AI script has actually sent data yet
        if isinstance(fetched_data, dict) and "current_density" in fetched_data:
            data = fetched_data
            status_indicator.success("Receiving Data Stream")
        else:
            status_indicator.warning("Backend up. Waiting for camera/AI...")
            
    except requests.exceptions.ConnectionError:
        status_indicator.error("Booting up backend... please wait.")
    except Exception as e:
        status_indicator.error(f"Error: {e}")

    # Display the metrics if we have valid data
    if data:
        density = data.get("current_density", 0)
        flow = data.get("current_flow", 0)
        pred_density = data.get("predicted_density", 0)
        pred_flow = data.get("predicted_flow", 0)
        risk = data.get("risk", False)
        timestamp = data.get("timestamp", "N/A")

        # Update session history
        if len(st.session_state.history) > 20:
            st.session_state.history.pop(0)
        st.session_state.history.append(data)

        # Row 1: Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Density", f"{density:.3f}")
        m2.metric("Current Flow", f"{flow:.3f}")
        m3.metric("Risk Level", "🚨 DANGER" if risk else "✅ SAFE", 
                   delta="High Risk" if risk else "Low Risk", delta_color="inverse")

        st.divider()

        # Row 2: Predictions and Time
        m4, m5, m6 = st.columns(3)
        m4.metric("Predicted Density (GRU)", f"{pred_density:.3f}")
        m5.metric("Predicted Flow (GRU)", f"{pred_flow:.3f}")
        m6.text(f"Last Backend Update:\n{timestamp}")

        # Row 3: Raw Data Table
        st.write("### Raw Backend Stream (Latest 20 Logs)")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------------
    # The Magic Loop: Wait 1 second, then reload the dashboard
    # ---------------------------------------------------------
    time.sleep(1)
    st.rerun()