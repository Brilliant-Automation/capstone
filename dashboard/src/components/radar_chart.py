from dash import dcc
import plotly.graph_objects as go
from utils.plot_config import format_plot
from utils.config import CHART_1_COLS, CHART_2_COLS, RATING_DESCRIPTIONS

# techdebt: more descriptive names
radar_chart_1 = dcc.Graph(
    id="radar-chart-1", 
    figure=go.Figure(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )

# techdebt: more descriptive names
radar_chart_2 = dcc.Graph(
    id="radar-chart-2", 
    figure=go.Figure(), 
    className="graph-container", 
    config={"displayModeBar": False}
    )

def update_radar_chart(df, chart_id):
    if df.empty:
        return go.Figure()
    
    cols = CHART_1_COLS if chart_id == 1 else CHART_2_COLS
    means = df[cols].mean()
    labels = [
        f"<span style='text-align:center; display:block'>{col.replace('_', ' ').title()}<br>{v:.1f}</span>"
        for col, v in zip(means.index, means.values)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=means.values,
        theta=labels,
        fill="toself",
        name="",
        hovertemplate=[
        f"{RATING_DESCRIPTIONS.get(col, '')}"
        for col in means.index
        ],
        hoverlabel=dict(
            namelength=0,
            font=dict(size=18)
        )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(tickfont=dict(size=16))
        ),
        showlegend=False,
        margin=dict(t=10, l=10, r=10, b=10)
    )

    return format_plot(fig)
