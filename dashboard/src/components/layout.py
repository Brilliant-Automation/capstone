from dash import html, dcc
import dash_bootstrap_components as dbc
from components.dropdowns import create_date_range_dropdown, device_dropdown, dual_sensor_dropdown
from components.radar_chart import radar_chart_1, radar_chart_2
from components.overlay_plot import rating_health_chart, frequency_chart
from components.signal_charts import signal_chart_sig_raw, signal_chart_env, signal_chart_sig_fft, signal_chart_env_fft
from components.header_timestamp import header_timestamp

def summary_view():
    return[
        # Rating radar charts
        dbc.Row([
            dbc.Col(radar_chart_1),
            dbc.Col(radar_chart_2)
        ], className="graph-row"),

        # Rating overlay
        dbc.Row([
            dbc.Col(rating_health_chart, width=12)
        ], className="graph-row")
    ]

def signal_view():
    return [
        # Signal overlay
        dbc.Row([
            dbc.Col(frequency_chart)
        ], className="graph-row"),

        # Signal charts
        dbc.Row([
            dbc.Col(signal_chart_sig_raw),
            dbc.Col(signal_chart_env),
        ], className="graph-row"),
        dbc.Row([
            dbc.Col(signal_chart_sig_fft),
            dbc.Col(signal_chart_env_fft),
        ], className="graph-row")
    ]


def create_layout():
    return dbc.Container([
        # Timestamp
        dbc.Row([
            dbc.Col(html.Img(src="assets/logo.png", height="50px"), width=4),
            dbc.Col([header_timestamp()]),
        ]),

        # Device and Time Dropdowns
        dbc.Row([
            dbc.Col(device_dropdown, width=4, className="dropdown-input"),
            dbc.Col(create_date_range_dropdown("start"), width=4),
            dbc.Col(create_date_range_dropdown("end"), width=4),
        ]),

        # Tabs
        dbc.Row([
            dbc.Col([
                dcc.Tabs(
                    id="view-tabs",
                    value="summary",
                    children=[
                        dcc.Tab(label="Ratings Data", value="summary"),
                        dcc.Tab(label="Sensor Location Data", value="signal"),
                    ]
                )
            ])
        ]),

        # Rating/Location Dropdowns
        dbc.Row([
            dbc.Col(dual_sensor_dropdown, width=12, className="dropdown-input"),
        ]),

        # Graphs
        dbc.Row([
            dbc.Col([
                html.Div(id="summary-view", children=summary_view(), className="tab-content-container"),
                html.Div(id="signal-view", children=signal_view(), className="tab-content-container", style={"display": "none"})
            ])
        ])
    ], className="dash-container", fluid=True)