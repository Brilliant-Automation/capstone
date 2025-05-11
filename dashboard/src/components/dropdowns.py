from dash import dcc, html
import dash_bootstrap_components as dbc
from utils.data_loader import load_data, get_unique_locations

df = load_data()

DEVICE_OPTIONS = [
    {"label": dev, "value": dev}
    for dev in sorted(df["Device"].dropna().unique())
]
SENSOR_OPTIONS = [
    {"label": loc, "value": loc} for loc in get_unique_locations(df)
]

device_dropdown = html.Div([
    html.Label("Select Device"),
    dcc.Dropdown(
        id="device-dropdown",
        options=DEVICE_OPTIONS,
        value=DEVICE_OPTIONS[0]["value"],
        clearable=False
    )
])

sensor_dropdown = html.Div([
    html.Label("Select Sensor Locations"),
    dcc.Dropdown(
        id="sensor-dropdown",
        options=SENSOR_OPTIONS,
        value=[opt["value"] for opt in SENSOR_OPTIONS],
        multi=True
    )
])
