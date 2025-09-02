import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np
import time
import json
import io
from datetime import datetime, timedelta



# Page configuration
st.set_page_config(
    page_title="Battery Management System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .battery-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .battery-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .battery-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
    }

    .charging { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        animation: pulse-green 2s infinite;
    }

    .discharging { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        animation: pulse-red 2s infinite;
    }

    .idle { 
        background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 15px 35px rgba(17, 153, 142, 0.3); }
        50% { box-shadow: 0 15px 35px rgba(17, 153, 142, 0.6); }
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 15px 35px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 15px 35px rgba(255, 107, 107, 0.6); }
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: scale(1.05);
    }

    .status-charging { 
        color: #00ff88; 
        font-weight: bold; 
        text-shadow: 0 0 10px #00ff88;
        animation: glow-green 2s infinite;
    }

    .status-discharging { 
        color: #ff4757; 
        font-weight: bold; 
        text-shadow: 0 0 10px #ff4757;
        animation: glow-red 2s infinite;
    }

    .status-idle { 
        color: #70a1ff; 
        font-weight: bold; 
        text-shadow: 0 0 10px #70a1ff;
    }

    @keyframes glow-green {
        0%, 100% { text-shadow: 0 0 10px #00ff88; }
        50% { text-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88; }
    }

    @keyframes glow-red {
        0%, 100% { text-shadow: 0 0 10px #ff4757; }
        50% { text-shadow: 0 0 20px #ff4757, 0 0 30px #ff4757; }
    }

    .control-panel {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }

    .battery-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .simulation-controls {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class BatterySimulator:
    def __init__(self):
        self.states = ["charging", "idle", "discharging", "idle"]
        self.state_icons = {
            "charging": "🔋⚡",
            "discharging": "🔋📱",
            "idle": "🔋💤"
        }
        self.state_colors = {
            "charging": "#00ff88",
            "discharging": "#ff4757",
            "idle": "#70a1ff"
        }
        self.process_types = {
            "CCV": "Constant Current Voltage (Charging)",
            "CCCV": "Constant Current Constant Voltage (Charging)",
            "CCD": "Constant Current Discharge",
            "IDLE": "Idle State",
            "PULSE": "Pulse Test",
            "REST": "Rest Period"
        }

    def get_battery_properties(self, cell_type):
        """Get battery properties based on chemistry"""
        properties = {
            "lfp": {
                "nominal_voltage": 3.2,
                "min_voltage": 2.8,
                "max_voltage": 3.6,
                "nominal_capacity": 100,
                "internal_resistance": 0.05
            },
            "nmc": {
                "nominal_voltage": 3.6,
                "min_voltage": 3.0,
                "max_voltage": 4.2,
                "nominal_capacity": 80,
                "internal_resistance": 0.08
            }
        }
        return properties.get(cell_type, properties["lfp"])

    def simulate_battery_behavior(self, battery_data, current, state, time_step=1):
        """Simulate realistic battery behavior"""
        props = self.get_battery_properties(battery_data["type"])

        # Current SOC
        current_soc = battery_data.get("soc", 50)
        current_voltage = battery_data.get("voltage", props["nominal_voltage"])
        current_temp = battery_data.get("temperature", 25)

        if state == "charging":
            # Charging behavior
            if current > 0:
                # SOC increases
                soc_change = (current * time_step) / props["nominal_capacity"] * 100
                new_soc = min(100, current_soc + soc_change * 0.1)

                # Voltage increases with SOC
                voltage_change = (new_soc - current_soc) * 0.01
                new_voltage = min(props["max_voltage"], current_voltage + voltage_change)

                # Temperature increases slightly during charging
                temp_change = current * 0.05 + random.uniform(-0.5, 1.0)
                new_temp = current_temp + temp_change
            else:
                new_soc = current_soc
                new_voltage = current_voltage
                new_temp = current_temp

        elif state == "discharging":
            # Discharging behavior
            if current > 0:
                # SOC decreases
                soc_change = (current * time_step) / props["nominal_capacity"] * 100
                new_soc = max(0, current_soc - soc_change * 0.1)

                # Voltage decreases with load
                voltage_drop = current * props["internal_resistance"]
                new_voltage = max(props["min_voltage"], current_voltage - voltage_drop - random.uniform(0, 0.02))

                # Temperature increases with load
                temp_change = current * 0.08 + random.uniform(-0.2, 1.5)
                new_temp = current_temp + temp_change
            else:
                new_soc = current_soc
                new_voltage = current_voltage
                new_temp = current_temp

        else:  # idle
            # Idle behavior - minimal changes
            new_soc = current_soc - random.uniform(0, 0.1)  # Self-discharge
            new_voltage = current_voltage + random.uniform(-0.01, 0.01)  # Small variations
            new_temp = current_temp + random.uniform(-1, 1)  # Temperature drift

        # Calculate power
        power = new_voltage * current if state != "idle" else 0

        return {
            "soc": round(max(0, min(100, new_soc)), 2),
            "voltage": round(max(props["min_voltage"], min(props["max_voltage"], new_voltage)), 3),
            "temperature": round(max(15, min(80, new_temp)), 1),
            "power": round(power, 2),
            "current": round(current, 2),
            "state": state
        }

    def simulate_process_step(self, battery_data, process_step, step_time):
        """Simulate battery behavior based on process step"""
        props = self.get_battery_properties(battery_data["type"])

        current_soc = battery_data.get("soc", 50)
        current_voltage = battery_data.get("voltage", props["nominal_voltage"])
        current_temp = battery_data.get("temperature", 25)

        process_type = process_step["type"]
        current = process_step["current"]
        target_voltage = process_step.get("target_voltage", props["max_voltage"])

        if process_type in ["CCV", "CCCV"]:
            # Charging process
            if current > 0:
                # SOC increases based on current and time
                soc_change = (current * step_time) / (props["nominal_capacity"] * 3600) * 100
                new_soc = min(100, current_soc + soc_change + random.uniform(-0.1, 0.1))

                # Voltage follows charging curve
                if process_type == "CCCV" and current_voltage >= target_voltage:
                    # CV phase - voltage constant, current decreases
                    new_voltage = target_voltage
                    actual_current = current * (1 - new_soc / 100) * 0.5  # Current taper
                else:
                    # CC phase - current constant, voltage increases
                    voltage_change = (new_soc - current_soc) * 0.015 + random.uniform(-0.01, 0.01)
                    new_voltage = min(props["max_voltage"], current_voltage + voltage_change)
                    actual_current = current

                # Temperature rise during charging
                temp_change = abs(actual_current) * 0.1 + random.uniform(-0.5, 1.5)
                new_temp = min(60, current_temp + temp_change * step_time / 60)

                state = "charging"
            else:
                new_soc, new_voltage, new_temp, actual_current, state = current_soc, current_voltage, current_temp, 0, "idle"

        elif process_type == "CCD":
            # Discharge process
            if current > 0:
                # SOC decreases
                soc_change = (current * step_time) / (props["nominal_capacity"] * 3600) * 100
                new_soc = max(0, current_soc - soc_change + random.uniform(-0.1, 0.1))

                # Voltage drops with discharge
                voltage_drop = current * props["internal_resistance"] + (100 - new_soc) * 0.005
                new_voltage = max(props["min_voltage"], current_voltage - voltage_drop - random.uniform(0, 0.02))

                # Temperature rise during discharge
                temp_change = current * 0.15 + random.uniform(-0.3, 2.0)
                new_temp = min(70, current_temp + temp_change * step_time / 60)

                actual_current = current
                state = "discharging"
            else:
                new_soc, new_voltage, new_temp, actual_current, state = current_soc, current_voltage, current_temp, 0, "idle"

        elif process_type in ["IDLE", "REST"]:
            # Idle/Rest state
            new_soc = max(0, current_soc - random.uniform(0, 0.05))  # Self discharge
            new_voltage = current_voltage + random.uniform(-0.005, 0.005)  # Small drift
            new_temp = current_temp * 0.99 + 25 * 0.01 + random.uniform(-0.5, 0.5)  # Cool towards ambient
            actual_current = 0
            state = "idle"

        elif process_type == "PULSE":
            # Pulse test - alternating current
            pulse_current = current if (step_time % 2) == 0 else -current * 0.5
            if pulse_current > 0:
                new_soc = min(100, current_soc + random.uniform(0, 0.1))
                new_voltage = min(props["max_voltage"], current_voltage + random.uniform(0, 0.05))
                state = "charging"
            else:
                new_soc = max(0, current_soc - random.uniform(0, 0.1))
                new_voltage = max(props["min_voltage"], current_voltage - random.uniform(0, 0.05))
                state = "discharging"
            new_temp = current_temp + random.uniform(-1, 2)
            actual_current = abs(pulse_current)
        else:
            new_soc, new_voltage, new_temp, actual_current, state = current_soc, current_voltage, current_temp, 0, "idle"

        # Calculate power
        power = new_voltage * actual_current

        return {
            "soc": round(max(0, min(100, new_soc)), 2),
            "voltage": round(max(props["min_voltage"], min(props["max_voltage"], new_voltage)), 3),
            "temperature": round(max(15, min(80, new_temp)), 1),
            "power": round(power, 2),
            "current": round(actual_current, 2),
            "state": state,
            "process_type": process_type
        }


def create_process_designer():
    """Create process design interface"""
    st.markdown("### 🛠️ Battery Test Process Designer")

    if 'current_process' not in st.session_state:
        st.session_state.current_process = []

    if 'saved_processes' not in st.session_state:
        st.session_state.saved_processes = {}

    # Process builder
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Add Process Step")

        step_col1, step_col2, step_col3 = st.columns(3)

        with step_col1:
            process_type = st.selectbox(
                "Process Type",
                ["CCV", "CCCV", "CCD", "IDLE", "PULSE", "REST"],
                help="Select the type of battery test process"
            )

        with step_col2:
            current_value = st.number_input(
                "Current (A)",
                min_value=0.0, max_value=50.0, value=5.0, step=0.1
            )

        with step_col3:
            duration_minutes = st.number_input(
                "Duration (minutes)",
                min_value=1, max_value=1440, value=60, step=1
            )

        # Additional parameters based on process type
        if process_type in ["CCV", "CCCV"]:
            target_voltage = st.number_input(
                "Target Voltage (V)",
                min_value=3.0, max_value=4.5, value=4.2, step=0.1
            )
        else:
            target_voltage = 4.2

        # Add step button
        if st.button("➕ Add Step"):
            step = {
                "type": process_type,
                "current": current_value,
                "duration": duration_minutes,
                "target_voltage": target_voltage,
                "description": f"{process_type} @ {current_value}A for {duration_minutes}min"
            }
            st.session_state.current_process.append(step)
            st.success(f"Added {process_type} step!")

    with col2:
        # Process management
        st.markdown("#### Process Management")

        process_name = st.text_input("Process Name", value="Custom Process")

        col_save, col_load = st.columns(2)

        with col_save:
            if st.button("💾 Save Process") and st.session_state.current_process:
                st.session_state.saved_processes[process_name] = st.session_state.current_process.copy()
                st.success(f"Saved '{process_name}'!")

        with col_load:
            if st.session_state.saved_processes:
                selected_process = st.selectbox(
                    "Load Saved Process",
                    [""] + list(st.session_state.saved_processes.keys())
                )
                if st.button("📂 Load") and selected_process:
                    st.session_state.current_process = st.session_state.saved_processes[selected_process].copy()
                    st.success(f"Loaded '{selected_process}'!")

        if st.button("🗑️ Clear Process"):
            st.session_state.current_process = []
            st.success("Process cleared!")

    # Display current process
    if st.session_state.current_process:
        st.markdown("#### Current Process Steps")

        total_time = sum(step["duration"] for step in st.session_state.current_process)
        st.info(f"Total Process Time: {total_time} minutes ({total_time / 60:.1f} hours)")

        # Process timeline visualization
        fig = go.Figure()

        cumulative_time = 0
        colors = {"CCV": "#00ff88", "CCCV": "#32cd32", "CCD": "#ff4757",
                  "IDLE": "#70a1ff", "PULSE": "#ffa502", "REST": "#747d8c"}

        for i, step in enumerate(st.session_state.current_process):
            fig.add_trace(go.Scatter(
                x=[cumulative_time, cumulative_time + step["duration"]],
                y=[step["current"], step["current"]],
                mode='lines',
                line=dict(color=colors.get(step["type"], "#70a1ff"), width=8),
                name=f"Step {i + 1}: {step['type']}",
                hovertemplate=f"<b>{step['type']}</b><br>Current: {step['current']}A<br>Duration: {step['duration']}min<extra></extra>"
            ))
            cumulative_time += step["duration"]

        fig.update_layout(
            title="Process Timeline",
            xaxis_title="Time (minutes)",
            yaxis_title="Current (A)",
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Process steps table
        for i, step in enumerate(st.session_state.current_process):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**Step {i + 1}:** {step['description']}")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.current_process.pop(i)
                    st.rerun()

    return st.session_state.current_process


def create_battery_analysis(battery_name, battery_data, history_data):
    """Create comprehensive battery analysis"""

    # Create subplots for detailed analysis
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Voltage Trend', 'State of Charge',
                        'Temperature Pattern', 'Power Output',
                        'Current Flow', 'Efficiency Analysis'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Extract history data
    timestamps = list(range(len(history_data)))
    voltages = [entry.get('voltage', 0) for entry in history_data]
    socs = [entry.get('soc', 0) for entry in history_data]
    temps = [entry.get('temperature', 0) for entry in history_data]
    powers = [entry.get('power', 0) for entry in history_data]
    currents = [entry.get('current', 0) for entry in history_data]
    states = [entry.get('state', 'idle') for entry in history_data]
    process_types = [entry.get('process_type', 'Unknown') for entry in history_data]

    # Color mapping for states
    colors = ['#00ff88' if s == 'charging' else '#ff4757' if s == 'discharging' else '#70a1ff' for s in states]

    # Voltage trend
    fig.add_trace(
        go.Scatter(x=timestamps, y=voltages, mode='lines+markers',
                   line=dict(color='#3498db', width=3),
                   marker=dict(color=colors, size=8),
                   name='Voltage'),
        row=1, col=1
    )

    # SOC trend
    fig.add_trace(
        go.Scatter(x=timestamps, y=socs, mode='lines+markers',
                   line=dict(color='#2ecc71', width=3),
                   marker=dict(color=colors, size=8),
                   fill='tonexty', fillcolor='rgba(46, 204, 113, 0.1)',
                   name='SOC'),
        row=1, col=2
    )

    # Temperature
    fig.add_trace(
        go.Scatter(x=timestamps, y=temps, mode='lines+markers',
                   line=dict(color='#e74c3c', width=3),
                   marker=dict(color=colors, size=8),
                   name='Temperature'),
        row=2, col=1
    )

    # Power
    fig.add_trace(
        go.Scatter(x=timestamps, y=powers, mode='lines+markers',
                   line=dict(color='#9b59b6', width=3),
                   marker=dict(color=colors, size=8),
                   name='Power'),
        row=2, col=2
    )

    # Current
    fig.add_trace(
        go.Scatter(x=timestamps, y=currents, mode='lines+markers',
                   line=dict(color='#f39c12', width=3),
                   marker=dict(color=colors, size=8),
                   name='Current'),
        row=3, col=1
    )

    # Efficiency (Power/Voltage ratio)
    efficiency = [p / v if v > 0 else 0 for p, v in zip(powers, voltages)]
    fig.add_trace(
        go.Scatter(x=timestamps, y=efficiency, mode='lines+markers',
                   line=dict(color='#1abc9c', width=3),
                   marker=dict(color=colors, size=8),
                   name='Efficiency'),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text=f"🔋 Comprehensive Analysis: {battery_name}",
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Update axes
    fig.update_xaxes(title_text="Time Steps", gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    return fig


def create_system_overview(batteries_data):
    """Create system overview dashboard"""

    if not batteries_data:
        return None

    # Prepare data
    names = list(batteries_data.keys())
    voltages = [data.get('voltage', 0) for data in batteries_data.values()]
    socs = [data.get('soc', 0) for data in batteries_data.values()]
    temps = [data.get('temperature', 0) for data in batteries_data.values()]
    powers = [data.get('power', 0) for data in batteries_data.values()]
    states = [data.get('state', 'idle') for data in batteries_data.values()]

    # Color mapping
    state_colors = {'charging': '#00ff88', 'discharging': '#ff4757', 'idle': '#70a1ff'}
    colors = [state_colors.get(state, '#70a1ff') for state in states]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Battery Voltages', 'State of Charge Distribution',
                        'Temperature vs Power', 'System Status'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )

    # Voltage bars
    fig.add_trace(
        go.Bar(x=names, y=voltages, marker_color=colors,
               text=[f"{v}V" for v in voltages], textposition='auto'),
        row=1, col=1
    )

    # SOC bars
    fig.add_trace(
        go.Bar(x=names, y=socs, marker_color=colors,
               text=[f"{s}%" for s in socs], textposition='auto'),
        row=1, col=2
    )

    # Temperature vs Power scatter
    fig.add_trace(
        go.Scatter(x=temps, y=powers, mode='markers+text',
                   marker=dict(size=20, color=colors),
                   text=names, textposition='top center'),
        row=2, col=1
    )

    # State distribution pie
    state_counts = {state: states.count(state) for state in set(states)}
    fig.add_trace(
        go.Pie(labels=list(state_counts.keys()), values=list(state_counts.values()),
               marker=dict(colors=[state_colors[state] for state in state_counts.keys()])),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False,
                      title_text="🔋 Battery System Overview")

    return fig


def export_data_to_csv():
    """Export battery data to CSV"""
    st.markdown("### 📥 Export Data to CSV")

    if not st.session_state.batteries or not st.session_state.history:
        st.warning("No data available to export. Run simulation first!")
        return

    # Data export options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Export Options")

        export_current_state = st.checkbox("Current Battery State", value=True)
        export_historical_data = st.checkbox("Historical Data", value=True)
        export_process_info = st.checkbox("Process Information", value=True)

        selected_batteries = st.multiselect(
            "Select Batteries to Export:",
            list(st.session_state.batteries.keys()),
            default=list(st.session_state.batteries.keys())
        )

    with col2:
        st.markdown("#### Data Preview")

        if selected_batteries:
            # Preview current state
            current_df = pd.DataFrame(
                {k: v for k, v in st.session_state.batteries.items() if k in selected_batteries}
            ).T
            st.dataframe(current_df.head(), use_container_width=True)

    # Export buttons
    if st.button("📊 Generate CSV Export", type="primary"):
        if not selected_batteries:
            st.error("Please select at least one battery!")
            return

        try:
            csv_data = []

            # Export current state data
            if export_current_state:
                for battery_name in selected_batteries:
                    battery_data = st.session_state.batteries[battery_name]
                    row = {
                        'Battery': battery_name,
                        'Type': battery_data.get('type', 'Unknown'),
                        'Voltage_V': battery_data.get('voltage', 0),
                        'SOC_%': battery_data.get('soc', 0),
                        'Temperature_C': battery_data.get('temperature', 0),
                        'Current_A': battery_data.get('current', 0),
                        'Power_W': battery_data.get('power', 0),
                        'State': battery_data.get('state', 'idle'),
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Data_Type': 'Current_State'
                    }
                    csv_data.append(row)

            # Export historical data
            if export_historical_data:
                for battery_name in selected_batteries:
                    if battery_name in st.session_state.history:
                        for i, entry in enumerate(st.session_state.history[battery_name]):
                            row = {
                                'Battery': battery_name,
                                'Type': st.session_state.batteries[battery_name].get('type', 'Unknown'),
                                'Voltage_V': entry.get('voltage', 0),
                                'SOC_%': entry.get('soc', 0),
                                'Temperature_C': entry.get('temperature', 0),
                                'Current_A': entry.get('current', 0),
                                'Power_W': entry.get('power', 0),
                                'State': entry.get('state', 'idle'),
                                'Process_Type': entry.get('process_type', 'Unknown'),
                                'Step': entry.get('step', i),
                                'Timestamp': entry.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                                'Data_Type': 'Historical'
                            }
                            csv_data.append(row)

            # Convert to DataFrame
            df = pd.DataFrame(csv_data)

            # Create CSV string
            csv_string = df.to_csv(index=False)

            # Provide download
            st.download_button(
                label="📥 Download CSV File",
                data=csv_string,
                file_name=f"battery_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            st.success(f"CSV generated with {len(df)} records!")

            # Show export summary
            st.markdown("#### Export Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Batteries", len(selected_batteries))
            with col3:
                current_count = len(df[df['Data_Type'] == 'Current_State'])
                historical_count = len(df[df['Data_Type'] == 'Historical'])
                st.metric("Current/Historical", f"{current_count}/{historical_count}")

        except Exception as e:
            st.error(f"Error generating CSV: {str(e)}")


def main():
    # Initialize session state
    if 'batteries' not in st.session_state:
        st.session_state.batteries = {}
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'history' not in st.session_state:
        st.session_state.history = {}
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'battery_processes' not in st.session_state:
        st.session_state.battery_processes = {}

    simulator = BatterySimulator()

    # Header
    st.markdown('<h1 class="main-header">🔋 Advanced Battery Management System</h1>',
                unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.markdown("## ⚙️ System Configuration")

    # Number of batteries
    num_batteries = st.sidebar.number_input("Number of Batteries",
                                            min_value=1, max_value=12, value=6)

    # Global current setting
    st.sidebar.markdown("### 🔌 Current Settings")
    global_current = st.sidebar.number_input("Global Current (A)",
                                             min_value=0.0, max_value=50.0,
                                             value=5.0, step=0.1)

    # Initialize batteries if not exists
    for i in range(num_batteries):
        battery_key = f"Battery {i + 1}"
        if battery_key not in st.session_state.batteries:
            battery_type = random.choice(["lfp", "nmc"])
            props = simulator.get_battery_properties(battery_type)

            st.session_state.batteries[battery_key] = {
                "type": battery_type,
                "voltage": props["nominal_voltage"],
                "soc": random.uniform(20, 80),
                "temperature": random.uniform(20, 30),
                "current": global_current,
                "power": 0,
                "state": "idle"
            }
            st.session_state.history[battery_key] = []

    # Simulation Controls
    st.sidebar.markdown("### 🎮 Simulation Controls")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("▶️ Start Simulation", type="primary"):
            st.session_state.simulation_running = True
            st.rerun()

    with col2:
        if st.button("⏹️ Stop Simulation"):
            st.session_state.simulation_running = False

    if st.sidebar.button("🔄 Reset All Batteries"):
        st.session_state.batteries = {}
        st.session_state.history = {}
        st.session_state.current_step = 0
        st.session_state.battery_processes = {}
        st.rerun()

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏠 Dashboard", "🛠️ Process Designer", "🔍 Battery Analysis", "📊 System Overview", "📥 Export Data"])

    with tab1:
        # Simulation status
        if st.session_state.simulation_running:
            st.markdown("""
            <div class="simulation-controls">
                <h3>🟢 Simulation Running</h3>
                <p>Batteries are cycling through processes or default states</p>
            </div>
            """, unsafe_allow_html=True)

            # Run simulation step
            for battery_key in st.session_state.batteries:
                battery_data = st.session_state.batteries[battery_key]

                if battery_key in st.session_state.battery_processes:
                    process_info = st.session_state.battery_processes[battery_key]

                    if process_info["process_active"] and process_info["process"]:
                        current_step_idx = process_info["current_step"]

                        if current_step_idx < len(process_info["process"]):
                            current_process_step = process_info["process"][current_step_idx]

                            # Check if current step is completed
                            elapsed_time = st.session_state.current_step - process_info["step_start_time"]

                            if elapsed_time >= current_process_step["duration"]:
                                # Move to next step
                                process_info["current_step"] += 1
                                process_info["step_start_time"] = st.session_state.current_step

                                if process_info["current_step"] >= len(process_info["process"]):
                                    process_info["process_active"] = False
                                    current_process_step = {"type": "IDLE", "current": 0, "target_voltage": 4.2}

                            # Simulate battery behavior based on process step
                            new_data = simulator.simulate_process_step(
                                battery_data, current_process_step, 1
                            )
                        else:
                            # Process completed, idle state
                            new_data = simulator.simulate_process_step(
                                battery_data, {"type": "IDLE", "current": 0, "target_voltage": 4.2}, 1
                            )
                    else:
                        # No active process, idle state
                        new_data = simulator.simulate_process_step(
                            battery_data, {"type": "IDLE", "current": 0, "target_voltage": 4.2}, 1
                        )
                else:
                    # No process assigned, use global current
                    battery_data["current"] = global_current
                    current_state_index = st.session_state.current_step % 4
                    current_state = simulator.states[current_state_index]

                    new_data = simulator.simulate_battery_behavior(
                        battery_data, global_current, current_state
                    )
                    new_data["process_type"] = f"Default_{current_state}"

                # Update battery data
                st.session_state.batteries[battery_key].update(new_data)

                # Store history with enhanced data
                history_entry = {
                    **new_data,
                    "step": st.session_state.current_step,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.history[battery_key].append(history_entry)

                # Limit history size
                if len(st.session_state.history[battery_key]) > 100:
                    st.session_state.history[battery_key].pop(0)

            st.session_state.current_step += 1
            time.sleep(0.1)  # Small delay
            st.rerun()

        else:
            st.markdown("""
            <div class="simulation-controls">
                <h3>⏸️ Simulation Paused</h3>
                <p>Click "Start Simulation" to begin battery state cycling</p>
            </div>
            """, unsafe_allow_html=True)

        # Display batteries
        st.markdown("### 🔋 Battery Grid")
        st.markdown("*Click 'Analyze' button on any battery for detailed analysis*")

        # Display batteries in grid
        cols_per_row = 3
        batteries_list = list(st.session_state.batteries.items())

        for i in range(0, len(batteries_list), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, (battery_name, battery_data) in enumerate(batteries_list[i:i + cols_per_row]):
                with cols[j]:
                    state = battery_data.get("state", "idle")
                    state_class = state
                    state_icon = simulator.state_icons.get(state, "🔋")

                    # Get process info if available
                    process_info = ""
                    if battery_name in st.session_state.battery_processes:
                        proc_info = st.session_state.battery_processes[battery_name]
                        if proc_info["process_active"]:
                            current_step = proc_info["current_step"]
                            if current_step < len(proc_info["process"]):
                                current_proc = proc_info["process"][current_step]
                                process_info = f"<br><strong>Process:</strong> {current_proc['type']}"

                    st.markdown(dedent(f"""
                    <div class="battery-card {state_class}">
                      <div class="battery-icon">{state_icon}</div>
                      <h3>{battery_name}</h3>
                      <div class="metric-card">
                        <strong>Type:</strong> {battery_data["type"].upper()}<br>
                        <strong>Voltage:</strong> {battery_data["voltage"]} V<br>
                        <strong>SOC:</strong> {battery_data["soc"]}%<br>
                        <strong>Temperature:</strong> {battery_data["temperature"]}°C<br>
                        <strong>Power:</strong> {battery_data["power"]} W<br>
                        <strong>State:</strong> <span class="status-{state}">{state.upper()}</span>
                        {process_info}
                      </div>
                    </div>
                    """), unsafe_allow_html=True)

                    if st.button(f"🔍 Analyze {battery_name}", key=f"analyze_{battery_name}"):
                        st.session_state.selected_battery = battery_name
                        st.success(f"Selected {battery_name}! Check Battery Analysis tab.")

    with tab2:
        # Process Designer Tab
        current_process = create_process_designer()

        if current_process:
            st.markdown("---")
            st.markdown("### 🎯 Apply Process to Batteries")

            if st.session_state.batteries:
                apply_to = st.multiselect(
                    "Select batteries to apply process:",
                    list(st.session_state.batteries.keys()),
                    default=list(st.session_state.batteries.keys())
                )

                if st.button("🚀 Apply Process to Selected Batteries", type="primary"):
                    if apply_to:
                        # Store process for selected batteries
                        for battery_name in apply_to:
                            st.session_state.battery_processes[battery_name] = {
                                "process": current_process.copy(),
                                "current_step": 0,
                                "step_start_time": st.session_state.current_step,
                                "process_active": True
                            }
                        st.success(f"Process applied to {len(apply_to)} batteries!")
                    else:
                        st.warning("Please select at least one battery!")
            else:
                st.info("Configure batteries first in the Dashboard tab.")

    with tab3:
        st.markdown("### 🔍 Individual Battery Analysis")

        if st.session_state.batteries:
            # Battery selection
            selected_battery = st.selectbox(
                "Select Battery for Analysis:",
                list(st.session_state.batteries.keys()),
                index=0
            )

            if selected_battery and selected_battery in st.session_state.history:
                battery_data = st.session_state.batteries[selected_battery]
                history_data = st.session_state.history[selected_battery]

                # Display current metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("⚡ Voltage", f"{battery_data['voltage']} V")
                with col2:
                    st.metric("🔋 SOC", f"{battery_data['soc']}%")
                with col3:
                    st.metric("🌡️ Temperature", f"{battery_data['temperature']}°C")
                with col4:
                    st.metric("⚙️ Power", f"{battery_data['power']} W")

                # Analysis charts
                if history_data:
                    analysis_fig = create_battery_analysis(selected_battery, battery_data, history_data)
                    st.plotly_chart(analysis_fig, use_container_width=True)

                    # Battery health assessment
                    st.markdown("### 🏥 Battery Health Assessment")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Health metrics
                        avg_voltage = np.mean([entry.get('voltage', 0) for entry in history_data[-10:]])
                        voltage_stability = np.std([entry.get('voltage', 0) for entry in history_data[-10:]])
                        temp_max = max([entry.get('temperature', 0) for entry in history_data[-10:]])

                        st.markdown(f"""
                        **Performance Metrics:**
                        - **Average Voltage:** {avg_voltage:.3f} V
                        - **Voltage Stability:** {voltage_stability:.3f} V
                        - **Max Temperature:** {temp_max:.1f}°C
                        - **Current State:** {battery_data['state'].upper()}
                        """)

                    with col2:
                        # Health score calculation
                        voltage_score = 100 if voltage_stability < 0.05 else max(0, 100 - voltage_stability * 1000)
                        temp_score = 100 if temp_max < 35 else max(0, 100 - (temp_max - 35) * 5)
                        overall_health = (voltage_score + temp_score) / 2

                        # Health indicator
                        health_color = "#00ff88" if overall_health > 80 else "#ffaa00" if overall_health > 60 else "#ff4757"

                        st.markdown(f"""
                        **Health Status:**
                        - **Voltage Health:** {voltage_score:.0f}%
                        - **Temperature Health:** {temp_score:.0f}%
                        - **Overall Health:** <span style="color: {health_color}; font-weight: bold;">{overall_health:.0f}%</span>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No historical data available. Run the simulation to collect data!")
        else:
            st.warning("No batteries configured. Please set up batteries first.")

    with tab4:
        st.markdown("### 📊 System Overview & Statistics")

        if st.session_state.batteries:
            # System metrics
            col1, col2, col3, col4 = st.columns(4)

            total_power = sum(battery.get('power', 0) for battery in st.session_state.batteries.values())
            avg_soc = np.mean([battery.get('soc', 0) for battery in st.session_state.batteries.values()])
            avg_temp = np.mean([battery.get('temperature', 0) for battery in st.session_state.batteries.values()])
            active_batteries = sum(
                1 for battery in st.session_state.batteries.values() if battery.get('state') != 'idle')

            with col1:
                st.metric("🔌 Total Power", f"{total_power:.1f} W")
            with col2:
                st.metric("⚡ Average SOC", f"{avg_soc:.1f}%")
            with col3:
                st.metric("🌡️ Average Temp", f"{avg_temp:.1f}°C")
            with col4:
                st.metric("🟢 Active Batteries", f"{active_batteries}/{len(st.session_state.batteries)}")

            # System overview charts
            overview_fig = create_system_overview(st.session_state.batteries)
            if overview_fig:
                st.plotly_chart(overview_fig, use_container_width=True)

            # Battery status table
            st.markdown("### 📋 Battery Status Table")

            battery_df = pd.DataFrame(st.session_state.batteries).T
            battery_df = battery_df[['type', 'voltage', 'soc', 'temperature', 'power', 'state']]
            battery_df.columns = ['Type', 'Voltage (V)', 'SOC (%)', 'Temperature (°C)', 'Power (W)', 'State']

            st.dataframe(battery_df, use_container_width=True)
        else:
            st.warning("No battery system configured.")

    with tab5:
        # Data Export Tab
        export_data_to_csv()


if __name__ == "__main__":

    main()


