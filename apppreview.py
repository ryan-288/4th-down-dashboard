# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Any, Dict

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update
import plotly.graph_objects as go

# Import API functions directly instead of using HTTP
try:
    from api import make_decision_logic, DecisionRequest, load_artifacts, artifacts
    # Load artifacts on startup
    try:
        load_artifacts()
    except Exception as e:
        print(f"Warning: Could not load artifacts on startup: {e}")
    USE_DIRECT_API = True
except ImportError:
    # Fallback to HTTP if API not available
    import requests
    API_URL = os.getenv("DECISION_API_URL", "http://localhost:8000")
    USE_DIRECT_API = False

# Modern color scheme
COLORS = {
    "go": {"primary": "#1a7f37", "light": "#e6f4ea", "dark": "#0d5a1f"},
    "fg": {"primary": "#ff6b35", "light": "#ffe6df", "dark": "#cc5529"},
    "punt": {"primary": "#4a90e2", "light": "#e6f0ff", "dark": "#2d5a8f"},
    "bg": "#f5f7fa",
    "text": "#2c3e50",
    "border": "#e1e8ed",
}

CARD_STYLE = {
    "border": f"2px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "12px",
    "backgroundColor": "#ffffff",
    "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
    "transition": "transform 0.2s, box-shadow 0.2s",
}


def compute_half_seconds(qtr: int, minutes_left: float) -> float:
    seconds_left = max(0.0, (minutes_left or 0) * 60.0)
    if qtr in (1, 3):
        return seconds_left + 900.0
    if qtr in (2, 4):
        return seconds_left
    # Overtime treated as standalone period
    return seconds_left


def pct(value: float) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def create_pie_chart(values: Dict[str, float], colors_list: list, title: str = "") -> go.Figure:
    """Create a pie chart from probability values."""
    labels = list(values.keys())
    vals = list(values.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=vals,
        hole=0.4,
        marker=dict(colors=colors_list),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=160,
    )
    return fig


def format_wpa_display(wpa: float) -> html.Div:
    """Format WPA with color coding."""
    if wpa is None:
        return html.Div("N/A", style={"fontSize": "12px", "color": "#999"})
    
    color = "#1a7f37" if wpa > 0 else "#d32f2f"
    sign = "+" if wpa > 0 else ""
    return html.Div(
        [
            html.Div("Win Probability Added", style={"fontSize": "10px", "color": "#666", "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Span(f"{sign}{wpa*100:.1f}%", style={"fontSize": "28px", "fontWeight": "bold", "color": color, "display": "block", "lineHeight": "1.2"}),
        ],
        style={"textAlign": "left", "marginBottom": "8px", "paddingBottom": "8px", "borderBottom": f"1px solid {COLORS['border']}"}
    )


def format_card(metrics: Dict[str, Any], mapping: Dict[str, str], card_type: str = "go") -> html.Div:
    """Create an enhanced card with visualizations."""
    items = []
    wpa_value = None
    
    for key, label in mapping.items():
        value = metrics.get(key)
        if value is None:
            display = "N/A"
        elif "prob" in key or "probability" in key:
            display = pct(value)
        elif "wpa" in key.lower():
            wpa_value = value
            # Don't add WPA to the list, we'll display it separately
            continue
        else:
            display = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        
        items.append(
            html.Li(
                [html.Strong(f"{label}: "), display],
                style={"marginBottom": "3px", "fontSize": "11px"}
            )
        )
    
    card_color = COLORS.get(card_type, COLORS["go"])
    
    return html.Div([
        format_wpa_display(wpa_value) if wpa_value is not None else html.Div(),
        html.Ul(items, style={"listStyle": "none", "padding": 0, "margin": 0, "marginTop": "8px"}),
    ])


app: Dash = dash.Dash(__name__)
app.title = "4th Down Decision Tool"

# Expose server for gunicorn
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                # Left Column - Inputs
        html.Div(
            [
                        html.H3("Game Situation", style={
                            "color": COLORS['text'],
                            "marginBottom": "12px",
                            "fontSize": "16px",
                            "borderBottom": f"2px solid {COLORS['border']}",
                            "paddingBottom": "6px"
                        }),
                html.Div(
                    [
                                html.Label("Yards to go", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                        dcc.Input(
                            id="yards-to-go",
                            type="number",
                            value=3,
                            min=0.5,
                            max=99,
                            step=0.5,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "14px",
                                        "boxSizing": "border-box"
                                    },
                                ),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Field position", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                                html.Div(
                                    [
                                        html.Button("Your side", id="side-own", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['go']['primary']}",
                                            "backgroundColor": COLORS['go']['light'],
                                            "color": COLORS['go']['primary'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                        html.Button("Opponent's side", id="side-opponent", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['border']}",
                                            "backgroundColor": "#ffffff",
                                            "color": COLORS['text'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                    ],
                                    style={"display": "flex", "gap": "4px", "width": "100%", "boxSizing": "border-box", "marginBottom": "8px"},
                                ),
                                dcc.Store(id="side-store", data="own"),
                                html.Label("Yardline (1-50)", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                        dcc.Input(
                            id="yard-line",
                            type="number",
                            value=40,
                            min=1,
                            max=50,
                            step=1,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "14px",
                                        "boxSizing": "border-box"
                                    },
                                ),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Quarter", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                html.Div(
                    [
                                        html.Button("1st", id="quarter-1", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['border']}",
                                            "backgroundColor": "#ffffff",
                                            "color": COLORS['text'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                        html.Button("2nd", id="quarter-2", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['border']}",
                                            "backgroundColor": "#ffffff",
                                            "color": COLORS['text'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                        html.Button("3rd", id="quarter-3", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['border']}",
                                            "backgroundColor": "#ffffff",
                                            "color": COLORS['text'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                        html.Button("4th", id="quarter-4", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['go']['primary']}",
                                            "backgroundColor": COLORS['go']['light'],
                                            "color": COLORS['go']['primary'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                        html.Button("OT", id="quarter-5", n_clicks=0, style={
                                            "flex": "1",
                                            "padding": "8px",
                                            "margin": "2px",
                                            "borderRadius": "4px",
                                            "border": f"2px solid {COLORS['border']}",
                                            "backgroundColor": "#ffffff",
                                            "color": COLORS['text'],
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "cursor": "pointer",
                                        }),
                                    ],
                                    style={"display": "flex", "gap": "4px", "width": "100%", "boxSizing": "border-box"},
                                ),
                                dcc.Store(id="quarter-store", data=4),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Time left (minutes)", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                        dcc.Input(
                            id="time-left",
                            type="number",
                            value=8,
                            min=0,
                            max=15,
                            step=0.5,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "14px",
                                        "boxSizing": "border-box"
                                    },
                                ),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Score differential", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                        dcc.Input(
                            id="score-diff",
                            type="number",
                            value=-4,
                            step=1,
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "fontSize": "14px",
                                        "boxSizing": "border-box"
                                    },
                                ),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Field goal distance (yards)", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                                dcc.Slider(
                            id="kicker-range",
                            min=20,
                            max=70,
                            step=1,
                                    value=57,
                                    marks={i: str(i) for i in range(20, 71, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                ),
                html.Div(
                    [
                                html.Label("Punt yards", style={"fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                                dcc.Slider(
                            id="punter-range",
                            min=20,
                            max=80,
                            step=1,
                                    value=45,
                                    marks={i: str(i) for i in range(20, 81, 15)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"marginBottom": "12px", "width": "100%", "boxSizing": "border-box"},
                        ),
                        html.Button(
                            "Analyze Decision",
                            id="analyze-button",
                            n_clicks=0,
                            style={
                                "backgroundColor": COLORS['go']['primary'],
                                "color": "white",
                                "border": "none",
                                "padding": "12px 24px",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "width": "100%",
                                "boxSizing": "border-box",
                                "transition": "background-color 0.3s",
                            }
                        ),
                        html.Div(id="error-message", style={"color": "crimson", "marginTop": "10px", "fontSize": "12px"}),
            ],
            style={
                        "width": "380px",
                        "backgroundColor": "#ffffff",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
                        "marginRight": "16px",
                        "height": "fit-content",
                        "position": "sticky",
                        "top": "16px",
                        "boxSizing": "border-box"
                    }
                ),
                # Right Column - Results
                html.Div(
                    [
        dcc.Loading(
            type="dot",
            children=html.Div(
                [
                    html.Div(
                        [
                                            html.H3("Go for it", style={
                                                "color": COLORS['go']['primary'],
                                                "marginBottom": "8px",
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                                "borderBottom": f"2px solid {COLORS['go']['primary']}",
                                                "paddingBottom": "6px"
                                            }),
                                            html.Div(
                                                [
                                                    html.Div(id="go-card", style={"flex": "1", "minWidth": "0", "paddingRight": "12px"}),
                                                    html.Div(id="go-chart", style={"flex": "2", "minWidth": "0", "maxWidth": "200px"}),
                                                ],
                                                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}
                                            ),
                                        ],
                                        style={**CARD_STYLE, "width": "calc(50% - 8px)", "boxSizing": "border-box", "minHeight": "280px"},
                    ),
                    html.Div(
                        [
                                            html.H3("Field Goal", style={
                                                "color": COLORS['fg']['primary'],
                                                "marginBottom": "8px",
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                                "borderBottom": f"2px solid {COLORS['fg']['primary']}",
                                                "paddingBottom": "6px"
                                            }),
                                            html.Div(
                                                [
                                                    html.Div(id="field-goal-card", style={"flex": "1", "minWidth": "0", "paddingRight": "12px"}),
                                                    html.Div(id="fg-chart", style={"flex": "2", "minWidth": "0", "maxWidth": "200px"}),
                                                ],
                                                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}
                                            ),
                                        ],
                                        style={**CARD_STYLE, "width": "calc(50% - 8px)", "boxSizing": "border-box", "minHeight": "280px"},
                    ),
                    html.Div(
                        [
                                            html.H3("Punt", style={
                                                "color": COLORS['punt']['primary'],
                                                "marginBottom": "8px",
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                                "borderBottom": f"2px solid {COLORS['punt']['primary']}",
                                                "paddingBottom": "6px"
                                            }),
                                            html.Div(
                                                [
                                                    html.Div(id="punt-card", style={"flex": "1", "minWidth": "0", "paddingRight": "12px"}),
                                                    html.Div(id="punt-chart", style={"flex": "2", "minWidth": "0", "maxWidth": "200px"}),
                                                ],
                                                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"}
                                            ),
                                        ],
                                        style={**CARD_STYLE, "width": "calc(50% - 8px)", "boxSizing": "border-box", "minHeight": "280px"},
        ),
        html.Div(
            [
                                            html.H3("Recommendation", style={
                                                "color": COLORS['text'],
                                                "fontSize": "18px",
                                                "fontWeight": "700",
                                                "marginBottom": "12px"
                                            }),
                html.Div(
                    id="recommendation-text",
                    style={
                                                    "fontSize": "20px",
                        "fontWeight": "bold",
                                                    "color": COLORS['go']['primary'],
                                                    "marginTop": "4px",
                                                    "padding": "12px",
                                                    "backgroundColor": COLORS['go']['light'],
                                                    "borderRadius": "6px",
                                                    "textAlign": "center",
                                                    "border": f"2px solid {COLORS['go']['primary']}",
                                                    "minHeight": "60px"
                    },
                ),
            ],
                                        style={**CARD_STYLE, "width": "calc(50% - 8px)", "boxSizing": "border-box", "minHeight": "280px"},
                                    ),
                                ],
                                style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "width": "100%", "boxSizing": "border-box"},
                            ),
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "flex-start",
                "border": "none",
                "outline": "none"
            }
        ),
    ],
    style={
        "maxWidth": "100%",
        "margin": "0",
        "padding": "16px",
        "backgroundColor": COLORS['bg'],
        "minHeight": "100vh",
        "width": "100%",
        "boxSizing": "border-box",
    },
)

# Set body background color
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f5f7fa;
                margin: 0;
                padding: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


@app.callback(
    Output("quarter-store", "data"),
    Output("quarter-1", "style"),
    Output("quarter-2", "style"),
    Output("quarter-3", "style"),
    Output("quarter-4", "style"),
    Output("quarter-5", "style"),
    Input("quarter-1", "n_clicks"),
    Input("quarter-2", "n_clicks"),
    Input("quarter-3", "n_clicks"),
    Input("quarter-4", "n_clicks"),
    Input("quarter-5", "n_clicks"),
    State("quarter-store", "data"),
)
def update_quarter(click1, click2, click3, click4, click5, current_quarter):
    ctx = dash.callback_context
    if not ctx.triggered:
        quarter = current_quarter or 4
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        quarter = int(button_id.split("-")[1])
    
    base_style = {
        "flex": "1",
        "padding": "8px",
        "margin": "2px",
        "borderRadius": "4px",
        "border": f"2px solid {COLORS['border']}",
        "backgroundColor": "#ffffff",
        "color": COLORS['text'],
        "fontSize": "14px",
        "fontWeight": "600",
        "cursor": "pointer",
    }
    
    active_style = {
        **base_style,
        "border": f"2px solid {COLORS['go']['primary']}",
        "backgroundColor": COLORS['go']['light'],
        "color": COLORS['go']['primary'],
    }
    
    styles = [
        active_style if quarter == 1 else base_style,
        active_style if quarter == 2 else base_style,
        active_style if quarter == 3 else base_style,
        active_style if quarter == 4 else base_style,
        active_style if quarter == 5 else base_style,
    ]
    
    return quarter, styles[0], styles[1], styles[2], styles[3], styles[4]


@app.callback(
    Output("side-store", "data"),
    Output("side-own", "style"),
    Output("side-opponent", "style"),
    Input("side-own", "n_clicks"),
    Input("side-opponent", "n_clicks"),
    State("side-store", "data"),
)
def update_side(click_own, click_opponent, current_side):
    ctx = dash.callback_context
    if not ctx.triggered:
        side = current_side or "own"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        side = "own" if button_id == "side-own" else "opponent"
    
    base_style = {
        "flex": "1",
        "padding": "8px",
        "margin": "2px",
        "borderRadius": "4px",
        "border": f"2px solid {COLORS['border']}",
        "backgroundColor": "#ffffff",
        "color": COLORS['text'],
        "fontSize": "14px",
        "fontWeight": "600",
        "cursor": "pointer",
    }
    
    active_style = {
        **base_style,
        "border": f"2px solid {COLORS['go']['primary']}",
        "backgroundColor": COLORS['go']['light'],
        "color": COLORS['go']['primary'],
    }
    
    own_style = active_style if side == "own" else base_style
    opponent_style = active_style if side == "opponent" else base_style
    
    return side, own_style, opponent_style


@app.callback(
    Output("go-card", "children"),
    Output("field-goal-card", "children"),
    Output("punt-card", "children"),
    Output("go-chart", "children"),
    Output("fg-chart", "children"),
    Output("punt-chart", "children"),
    Output("recommendation-text", "children"),
    Output("error-message", "children"),
    Input("analyze-button", "n_clicks"),
    State("yards-to-go", "value"),
    State("yard-line", "value"),
    State("side-store", "data"),
    State("quarter-store", "data"),
    State("time-left", "value"),
    State("score-diff", "value"),
    State("kicker-range", "value"),
    State("punter-range", "value"),
    prevent_initial_call=True,
)
def analyze_decision(
    n_clicks,
    ydstogo,
    yardline,
    side,
    quarter,
    time_left,
    score_diff,
    kicker_range,
    punter_range,
):
    inputs = [
        ydstogo,
        yardline,
        side,
        quarter,
        time_left,
        score_diff,
        kicker_range,
        punter_range,
    ]
    if any(value is None for value in inputs):
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            "Please fill in all inputs before running the analysis.",
        )

    # Convert yardline to yardline_100 based on side
    # If "own" side: yardline_100 = 100 - yardline (e.g., own 40 = 60 yards to end zone)
    # If "opponent" side: yardline_100 = yardline (e.g., opponent 40 = 40 yards to end zone)
    if side == "own":
        yardline_100 = 100 - float(yardline)
    else:
        yardline_100 = float(yardline)

    # Prepare request payload
    payload_dict = {
        "ydstogo": float(ydstogo),
        "yardline_100": yardline_100,
        "qtr": int(quarter),
        "half_seconds_remaining": compute_half_seconds(int(quarter), float(time_left)),
        "score_differential": float(score_diff),
        "gross_punt_yards": float(punter_range),
        "kick_distance": float(kicker_range),
    }

    try:
        if USE_DIRECT_API:
            # Call API functions directly
            # Ensure artifacts are loaded
            if not artifacts.loaded():
                load_artifacts()
            request_obj = DecisionRequest(**payload_dict)
            data = make_decision_logic(request_obj)
        else:
            # Fallback to HTTP
            if not API_URL:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "API URL not configured.",
                )
            response = requests.post(
                f"{API_URL.rstrip('/')}/decision", json=payload_dict, timeout=8
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update, f"Error: {exc}")

    go_metrics = data.get("go_for_it", {})
    fg_metrics = data.get("field_goal", {})
    punt_metrics = data.get("punt", {})
    recommendation = data.get("recommendation", {})

    # Create pie charts
    go_conv_prob = go_metrics.get("conversion_prob", 0)
    go_chart_fig = create_pie_chart(
        {"Success": go_conv_prob, "Fail": 1 - go_conv_prob},
        [COLORS['go']['primary'], "#d32f2f"],
        ""
    )
    
    fg_make_prob = fg_metrics.get("make_prob", 0)
    fg_chart_fig = create_pie_chart(
        {"Made": fg_make_prob, "Missed": 1 - fg_make_prob},
        [COLORS['fg']['primary'], "#d32f2f"],
        ""
    )
    
    opp_td = max(0, punt_metrics.get("opp_td_prob", 0))
    opp_fg = max(0, punt_metrics.get("opp_fg_prob", 0))
    opp_no = max(0, punt_metrics.get("opp_no_score_prob", 0))
    
    # Normalize to ensure they sum to 1.0 for the pie chart and display
    total = opp_td + opp_fg + opp_no
    if total > 0:
        opp_td_norm = opp_td / total
        opp_fg_norm = opp_fg / total
        opp_no_norm = opp_no / total
    else:
        opp_td_norm = opp_fg_norm = opp_no_norm = 0
    
    punt_chart_fig = create_pie_chart(
        {"TD": opp_td_norm, "FG": opp_fg_norm, "No Score": opp_no_norm},
        ["#d32f2f", COLORS['fg']['primary'], COLORS['punt']['primary']],
        ""
    )

    go_card = format_card(
        go_metrics,
        {
            "conversion_prob": "Conversion probability",
            "expected_wpa": "Expected WPA",
            "expected_epa": "Expected EPA",
        },
        "go"
    )
    fg_card = format_card(
        fg_metrics,
        {
            "make_prob": "Make probability",
            "expected_wpa": "Expected WPA",
            "expected_epa": "Expected EPA",
        },
        "fg"
    )
    # Use normalized values for punt card so they match the chart
    punt_card = format_card(
        {**punt_metrics, "expected_wpa": punt_metrics.get("wpa"), 
         "opp_td_prob": opp_td_norm, "opp_fg_prob": opp_fg_norm, 
         "opp_no_score_prob": opp_no_norm},
        {
            "expected_wpa": "Expected WPA",
            "epa": "EPA",
            "opp_td_prob": "Opponent TD %",
            "opp_fg_prob": "Opponent FG %",
            "opp_no_score_prob": "Opponent no-score %",
        },
        "punt"
    )

    rec_play = recommendation.get("play", "N/A").replace("_", " ").title()
    rec_wpa = recommendation.get("expected_wpa", 0.0)
    rec_wpa_pct = rec_wpa * 100
    sign = "+" if rec_wpa > 0 else ""
    rec_text = html.Div([
        html.Div(rec_play, style={"fontSize": "24px", "marginBottom": "4px"}),
        html.Div(f"{sign}{rec_wpa_pct:.1f}% WPA", style={"fontSize": "14px", "color": "#666"}),
        html.Div(f"{sign}{rec_wpa_pct:.1f}% Δ Win%", style={"fontSize": "14px", "color": "#666"})
    ])

    return (
        go_card,
        fg_card,
        punt_card,
        dcc.Graph(figure=go_chart_fig, config={'displayModeBar': False}),
        dcc.Graph(figure=fg_chart_fig, config={'displayModeBar': False}),
        dcc.Graph(figure=punt_chart_fig, config={'displayModeBar': False}),
        rec_text,
        ""
    )


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8050))
    # Use production server if available, otherwise development server
    if os.getenv("RENDER") or os.getenv("PORT"):
        # Production: use gunicorn via command line
        # This file will be imported by gunicorn
        pass
    else:
        # Development: use built-in server
        app.run(debug=False, host="0.0.0.0", port=port)
