from dash import dcc
import plotly.graph_objs as go
from utils.plot_config import format_plot
from utils.colours import COLOUR_MAP

rating_health_chart = dcc.Graph(
    id="rating-health-chart",
    figure=go.Figure(),
    className="graph-container",
    config={"displayModeBar": False}
)

frequency_chart = dcc.Graph(
    id="frequency-chart",
    figure=go.Figure(),
    className="graph-container",
    config={"displayModeBar": False}
)

def create_overlay_figure(df, y_columns, y_label):
    traces = []
    
    for col in y_columns:
        for loc in df["location"].unique():
            subset = df[df["location"] == loc]
            traces.append(go.Scatter(
                x=subset["timestamp"],
                y=subset[col],
                mode="lines",
                name=f"{loc} - {col}",
                line=dict(color=COLOUR_MAP.get(loc))
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=10, b=40),
        hovermode="closest"
    )
    return format_plot(fig)
