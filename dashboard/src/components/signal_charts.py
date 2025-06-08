from dash import dcc
from components.signal_visualizer import SignalVisualizer
import plotly.graph_objects as go
from utils.config import FEATURES

STANDARD_NUMBER_OF_PLOTS = 4

def create_signal_chart(chart_id):
    """Create a signal chart component with the given ID"""
    return dcc.Graph(
        id=chart_id,
        figure=go.Figure(),
        className="signal-chart", 
        config={"displayModeBar": False}
    )

signal_chart_sig_raw = create_signal_chart("signal-chart-sig-raw")
signal_chart_sig_fft = create_signal_chart("signal-chart-sig-fft")
signal_chart_env = create_signal_chart("signal-chart-env")
signal_chart_env_fft = create_signal_chart("signal-chart-env-fft")

def update_signal_charts(df):
    # techdebt: config file where colnames can be put, currently fragile
    if df.empty or FEATURES['vibration_velocity_z'] not in df.columns:
        return [go.Figure()] * STANDARD_NUMBER_OF_PLOTS
    return SignalVisualizer(df).generate()