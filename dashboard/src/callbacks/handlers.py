from dash import Input, Output, State, dash, callback_context
from dash.exceptions import PreventUpdate
from utils.data_loader import load_data, get_unique_locations
from components.radar_chart import update_radar_chart
from components.overlay_plot import create_overlay_figure
from components.signal_charts import update_signal_charts
from components.constants import CHART_1_COLS, CHART_2_COLS
from utils.colours import COLOUR_EMOJI


def register_callbacks(app):
    @app.callback(
        [Output("ratings-dropdown-container", "style"),
         Output("locations-dropdown-container", "style")],
        Input("view-tabs", "value")
    )
    def toggle_dropdown_visibility(tab):
        """Toggle visibility between ratings and locations dropdowns based on tab"""
        if tab == "summary":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("rating-health-chart", "figure"),
        [
            Input("start-date", "date"),
            Input("end-date", "date"),
            Input("start-time", "value"),
            Input("end-time", "value"),
            Input("device-dropdown", "value"),
            Input("ratings-dropdown", "value")
        ],
        State("view-tabs", "value")
    )
    def update_health_chart(start_date, end_date, start_time, end_time, selected_device, selected_ratings, current_tab):
        """Update health chart only when in summary mode and inputs change"""
        if current_tab != "summary" or selected_ratings is None or selected_device is None:
            from plotly.graph_objects import Figure
            return Figure()
        
        start_datetime = f"{start_date} {start_time}" if start_date and start_time else None
        end_datetime = f"{end_date} {end_time}" if end_date and end_time else None
        
        df = load_data(selected_device, start_datetime, end_datetime)
        df = df[df["Device"] == selected_device]
        
        return create_overlay_figure(df, y_columns=selected_ratings, y_label="Rating Health")

    @app.callback(
        [
            Output("timestamp-header", "children"),
            Output("radar-chart-1", "figure"),
            Output("radar-chart-2", "figure"),
            Output("frequency-chart", "figure"),
            Output("signal-chart-sig-raw", "figure"),
            Output("signal-chart-sig-fft", "figure"),
            Output("signal-chart-env", "figure"),
            Output("signal-chart-env-fft", "figure")
        ],
        [
            Input("start-date", "date"),
            Input("end-date", "date"),
            Input("start-time", "value"),
            Input("end-time", "value"),
            Input("device-dropdown", "value"),
            Input("locations-dropdown", "value"),
            Input("view-tabs", "value")
        ]
    )
    def update_all_charts(start_date, end_date, start_time, end_time, selected_device, selected_locations, current_tab):
        if selected_locations is None or selected_device is None:
            from plotly.graph_objects import Figure
            empty_fig = Figure()
            return "No data loaded", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        start_datetime = f"{start_date} {start_time}" if start_date and start_time else None
        end_datetime = f"{end_date} {end_time}" if end_date and end_time else None

        df = load_data(selected_device, start_datetime, end_datetime)
        df = df[df["Device"] == selected_device]
        
        if current_tab == "signal":
            df = df[df["location"].isin(selected_locations)]

        if df.empty or "timestamp" not in df.columns:
            header_str = "No data loaded"
        else:
            start = df["timestamp"].min().strftime('%Y-%m-%d %H:%M:%S')
            end = df["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
            header_str = f"{start} - {end}"

        radar1 = update_radar_chart(df, chart_id=1)
        radar2 = update_radar_chart(df, chart_id=2)
        freq_fig = create_overlay_figure(df, y_columns=["High-Frequency Acceleration"], y_label="High-Frequency Acceleration (a.u.)")
        signal_figs = update_signal_charts(df)

        return header_str, radar1, radar2, freq_fig, *signal_figs

    @app.callback(
        Output("summary-view", "style"),
        Output("signal-view", "style"),
        Input("view-tabs", "value")
    )
    def toggle_tab_view(tab):
        if tab == "summary":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}