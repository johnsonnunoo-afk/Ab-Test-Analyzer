"""
A/B Test Analyzer — Split Decision
Conversion experiment analyzer with two-proportion z-test statistics.

Run:
    pip install dash plotly pandas scipy numpy
    python ab_test_analyzer.py

Then open http://localhost:8050
"""

import math
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from scipy import stats
import numpy as np

# ── Theme ──────────────────────────────────────────────────────────────────────

BG        = "#0A0C0F"
SURFACE   = "#111318"
CARD      = "#161B24"
BORDER    = "#1F2535"
TEXT      = "#E2E8F2"
MUTED     = "#5A6480"
DIM       = "#2E3650"
ACCENT    = "#4F8EF7"      # blue
GREEN     = "#22D18B"
RED       = "#F75A5A"
YELLOW    = "#F7C948"

FONT_DISPLAY = "'Syne', sans-serif"
FONT_MONO    = "'JetBrains Mono', monospace"
FONT_BODY    = "'DM Sans', sans-serif"

# ── Scenario presets ──────────────────────────────────────────────────────────

SCENARIOS = {
    "clear":  dict(va=10000, ca=400,  vb=10000, cb=510,  conf=95),
    "early":  dict(va=320,   ca=13,   vb=290,   cb=15,   conf=95),
    "under":  dict(va=10000, ca=510,  vb=10000, cb=400,  conf=95),
}

# ── Statistics engine ─────────────────────────────────────────────────────────

def compute_stats(va, ca, vb, cb, conf_pct):
    """Two-proportion z-test (pooled). Returns a dict of results."""
    if va <= 0 or vb <= 0:
        return None

    pa = ca / va if va > 0 else 0
    pb = cb / vb if vb > 0 else 0
    n  = va + vb
    p_pool = (ca + cb) / n if n > 0 else 0

    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1/va + 1/vb)) if p_pool not in (0, 1) else 1e-9
    z = (pb - pa) / se_pool if se_pool > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    alpha = 1 - conf_pct / 100
    z_crit = stats.norm.ppf(1 - alpha / 2)

    se_unpooled = math.sqrt(pa*(1-pa)/va + pb*(1-pb)/vb) if va > 0 and vb > 0 else 0
    ci_low  = (pb - pa) - z_crit * se_unpooled
    ci_high = (pb - pa) + z_crit * se_unpooled

    lift_rel = (pb - pa) / pa * 100 if pa > 0 else 0
    lift_abs = (pb - pa) * 100

    # Post-hoc power
    z_beta = abs(z) - z_crit
    power  = stats.norm.cdf(z_beta) * 100

    significant = p_value < alpha

    if not significant:
        verdict = "too_early" if (va < 1000 or vb < 1000) else "no_winner"
    elif pb > pa:
        verdict = "ship_b"
    else:
        verdict = "keep_a"

    return dict(
        pa=pa, pb=pb, z=z, p_value=p_value, alpha=alpha,
        ci_low=ci_low, ci_high=ci_high,
        lift_rel=lift_rel, lift_abs=lift_abs,
        power=power, significant=significant,
        verdict=verdict, z_crit=z_crit,
    )


def sample_size(baseline_pct, lift_pct, alpha=0.05, power=0.80):
    """Sample size per variant using normal approximation."""
    if baseline_pct <= 0 or lift_pct <= 0:
        return None
    p1 = baseline_pct / 100
    p2 = p1 * (1 + lift_pct / 100)
    if p2 >= 1:
        return None
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) +
         z_b * math.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p2 - p1)**2
    return math.ceil(n)


# ── UI helpers ────────────────────────────────────────────────────────────────

def labeled_input(label, id_, value, placeholder="", type_="number", min_=0):
    return html.Div([
        html.Label(label, style={
            "color": MUTED, "fontSize": "10px", "letterSpacing": "0.12em",
            "textTransform": "uppercase", "display": "block", "marginBottom": "6px",
        }),
        dcc.Input(
            id=id_, type=type_, value=value, placeholder=placeholder, min=min_,
            style={
                "width": "100%", "background": BG, "border": f"1px solid {BORDER}",
                "borderRadius": "6px", "color": TEXT, "padding": "10px 12px",
                "fontSize": "14px", "fontFamily": FONT_MONO, "outline": "none",
            },
            debounce=True,
        ),
    ], style={"marginBottom": "14px"})


def stat_chip(label, value, color=TEXT):
    return html.Div([
        html.Div(value, style={
            "color": color, "fontSize": "22px", "fontWeight": "700",
            "fontFamily": FONT_MONO, "lineHeight": "1",
        }),
        html.Div(label, style={
            "color": MUTED, "fontSize": "10px", "letterSpacing": "0.1em",
            "textTransform": "uppercase", "marginTop": "4px",
        }),
    ], style={
        "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "8px", "padding": "16px 20px", "flex": "1", "minWidth": "100px",
    })


def detail_row(label, value):
    return html.Div([
        html.Span(label, style={"color": MUTED, "fontSize": "12px"}),
        html.Span(value, style={"color": TEXT, "fontSize": "12px", "fontFamily": FONT_MONO}),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "padding": "8px 0", "borderBottom": f"1px solid {BORDER}22",
    })


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="A/B Test Analyzer — Split Decision")

app.index_string = """<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0A0C0F;color:#E2E8F2;font-family:'DM Sans',sans-serif}
    input[type=number]::-webkit-inner-spin-button{opacity:.3}
    input:focus{border-color:#4F8EF7 !important;box-shadow:0 0 0 3px #4F8EF720}
    ::-webkit-scrollbar{width:5px}
    ::-webkit-scrollbar-thumb{background:#1F2535;border-radius:3px}
  </style>
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>"""

app.layout = html.Div([

    # ── Header ──
    html.Div([
        html.Div([
            html.Div([
                html.Span("A/B", style={
                    "background": ACCENT, "color": BG, "fontWeight": "800",
                    "fontSize": "11px", "padding": "3px 7px", "borderRadius": "4px",
                    "fontFamily": FONT_MONO, "letterSpacing": "0.05em",
                }),
                html.Span("Split Decision", style={
                    "color": TEXT, "fontSize": "13px", "marginLeft": "10px",
                    "fontFamily": FONT_DISPLAY, "fontWeight": "700",
                }),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),

            html.H1("Should you ship\nVersion B?", style={
                "fontSize": "clamp(28px, 4vw, 48px)", "fontWeight": "800",
                "fontFamily": FONT_DISPLAY, "lineHeight": "1.1",
                "color": TEXT, "marginBottom": "14px", "whiteSpace": "pre-line",
            }),
            html.P(
                "Drop in your experiment numbers. Get conversion lift, p-value, "
                "confidence interval, and a deploy recommendation grounded in statistics — not vibes.",
                style={"color": MUTED, "fontSize": "14px", "lineHeight": "1.6",
                       "maxWidth": "480px", "marginBottom": "20px"}
            ),
            html.Div([
                html.Span("Hypothesis testing · two-proportion z-test",
                          style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_MONO}),
            ]),
        ], style={"flex": "1"}),

        # Scenario buttons
        html.Div([
            html.P("Try a scenario:", style={"color": MUTED, "fontSize": "11px",
                                              "marginBottom": "8px", "textTransform": "uppercase",
                                              "letterSpacing": "0.1em"}),
            html.Div([
                html.Button("Clear winner",   id="btn-clear", n_clicks=0),
                html.Button("Too early",      id="btn-early", n_clicks=0),
                html.Button("B underperforms",id="btn-under", n_clicks=0),
            ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
        ], style={"alignSelf": "flex-end"}),

    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "flex-start",
        "padding": "48px 48px 36px", "borderBottom": f"1px solid {BORDER}",
        "flexWrap": "wrap", "gap": "24px",
    }),

    # ── Main body ──
    html.Div([

        # Left column — inputs
        html.Div([

            # Variant A
            html.Div([
                html.Div([
                    html.Span("Variant A", style={"color": TEXT, "fontSize": "13px",
                                                   "fontWeight": "600", "fontFamily": FONT_DISPLAY}),
                    html.Span("Control", style={
                        "color": MUTED, "fontSize": "10px", "background": DIM,
                        "padding": "2px 8px", "borderRadius": "4px", "marginLeft": "8px",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
                html.Div(id="rate-a-display", style={
                    "fontFamily": FONT_MONO, "fontSize": "28px", "fontWeight": "600",
                    "color": ACCENT, "marginBottom": "16px",
                }),
                labeled_input("Visitors", "visitors-a", 10000, min_=1),
                labeled_input("Conversions", "conversions-a", 400, min_=0),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px", "marginBottom": "12px",
            }),

            # Variant B
            html.Div([
                html.Div([
                    html.Span("Variant B", style={"color": TEXT, "fontSize": "13px",
                                                   "fontWeight": "600", "fontFamily": FONT_DISPLAY}),
                    html.Span("Challenger", style={
                        "color": MUTED, "fontSize": "10px", "background": DIM,
                        "padding": "2px 8px", "borderRadius": "4px", "marginLeft": "8px",
                    }),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
                html.Div(id="rate-b-display", style={
                    "fontFamily": FONT_MONO, "fontSize": "28px", "fontWeight": "600",
                    "color": GREEN, "marginBottom": "16px",
                }),
                labeled_input("Visitors", "visitors-b", 10000, min_=1),
                labeled_input("Conversions", "conversions-b", 510, min_=0),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px", "marginBottom": "12px",
            }),

            # Confidence level
            html.Div([
                html.P("Confidence level", style={"color": MUTED, "fontSize": "10px",
                                                   "letterSpacing": "0.12em", "textTransform": "uppercase",
                                                   "marginBottom": "12px"}),
                dcc.RadioItems(
                    id="confidence",
                    options=[{"label": f"{v}%", "value": v} for v in [90, 95, 99]],
                    value=95,
                    inline=True,
                    style={"color": TEXT, "fontFamily": FONT_MONO, "fontSize": "13px"},
                    inputStyle={"marginRight": "6px", "accentColor": ACCENT},
                    labelStyle={"marginRight": "20px"},
                ),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px", "marginBottom": "12px",
            }),

            # Sample size calculator
            html.Div([
                html.H3("Sample size calculator", style={
                    "color": TEXT, "fontSize": "13px", "fontFamily": FONT_DISPLAY,
                    "fontWeight": "700", "marginBottom": "4px",
                }),
                html.P("How many visitors per variant to detect a real effect?",
                       style={"color": MUTED, "fontSize": "11px", "marginBottom": "16px"}),
                labeled_input("Baseline rate (%)", "ss-baseline", 4.0, min_=0.01),
                labeled_input("Min. detectable lift (%)", "ss-lift", 20.0, min_=0.01),
                html.Div([
                    html.Div(id="sample-size-output", style={
                        "fontFamily": FONT_MONO, "fontSize": "28px", "fontWeight": "700",
                        "color": ACCENT,
                    }),
                    html.P("visitors per variant", style={"color": MUTED, "fontSize": "11px", "marginTop": "4px"}),
                    html.P("α = 0.05 · power = 80%", style={"color": DIM, "fontSize": "10px",
                                                              "fontFamily": FONT_MONO, "marginTop": "2px"}),
                ]),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px",
            }),

        ], style={"width": "300px", "flexShrink": "0"}),

        # Right column — results
        html.Div([

            # Verdict banner
            html.Div(id="verdict-banner", style={"marginBottom": "16px"}),

            # Stat chips row
            html.Div(id="stat-chips", style={
                "display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "16px",
            }),

            # Distribution chart
            html.Div([
                html.P("Sampling distribution of the difference (B − A)",
                       style={"color": MUTED, "fontSize": "10px", "letterSpacing": "0.1em",
                              "textTransform": "uppercase", "marginBottom": "12px"}),
                dcc.Graph(id="dist-chart", config={"displayModeBar": False},
                          style={"height": "220px"}),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px", "marginBottom": "16px",
            }),

            # Conversion rate comparison chart
            html.Div([
                html.P("Conversion rate comparison",
                       style={"color": MUTED, "fontSize": "10px", "letterSpacing": "0.1em",
                              "textTransform": "uppercase", "marginBottom": "12px"}),
                dcc.Graph(id="rate-chart", config={"displayModeBar": False},
                          style={"height": "160px"}),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px", "marginBottom": "16px",
            }),

            # Statistical detail
            html.Div([
                html.H3("Statistical detail", style={
                    "color": TEXT, "fontSize": "13px", "fontFamily": FONT_DISPLAY,
                    "fontWeight": "700", "marginBottom": "16px",
                }),
                html.Div(id="stat-detail"),
            ], style={
                "background": CARD, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px",
            }),

        ], style={"flex": "1", "minWidth": "0"}),

    ], style={"display": "flex", "gap": "16px", "padding": "32px 48px", "alignItems": "flex-start"}),

    # Footer
    html.Div([
        html.Span("Built for honest decisions under uncertainty · α · β · p",
                  style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_MONO}),
    ], style={"textAlign": "center", "padding": "16px 0 32px",
               "borderTop": f"1px solid {BORDER}"}),

], style={"minHeight": "100vh", "background": BG})


# ── Scenario button styling ───────────────────────────────────────────────────
for btn_id in ["btn-clear", "btn-early", "btn-under"]:
    app.layout.children[0].children[1].children[1].children  # just ensure layout built

BTN_STYLE = {
    "background": "transparent", "color": MUTED,
    "border": f"1px solid {BORDER}", "borderRadius": "6px",
    "padding": "6px 14px", "cursor": "pointer",
    "fontSize": "11px", "fontFamily": FONT_MONO,
}


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("visitors-a",     "value"),
    Output("conversions-a",  "value"),
    Output("visitors-b",     "value"),
    Output("conversions-b",  "value"),
    Output("confidence",     "value"),
    Input("btn-clear", "n_clicks"),
    Input("btn-early", "n_clicks"),
    Input("btn-under", "n_clicks"),
    prevent_initial_call=True,
)
def load_scenario(n1, n2, n3):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    key = {"btn-clear": "clear", "btn-early": "early", "btn-under": "under"}[btn]
    s = SCENARIOS[key]
    return s["va"], s["ca"], s["vb"], s["cb"], s["conf"]


@app.callback(
    Output("rate-a-display",  "children"),
    Output("rate-b-display",  "children"),
    Output("verdict-banner",  "children"),
    Output("stat-chips",      "children"),
    Output("dist-chart",      "figure"),
    Output("rate-chart",      "figure"),
    Output("stat-detail",     "children"),
    Output("sample-size-output", "children"),
    Input("visitors-a",    "value"),
    Input("conversions-a", "value"),
    Input("visitors-b",    "value"),
    Input("conversions-b", "value"),
    Input("confidence",    "value"),
    Input("ss-baseline",   "value"),
    Input("ss-lift",       "value"),
)
def update_all(va, ca, vb, cb, conf, ss_base, ss_lift):

    # ── defaults / guards ──
    va = int(va or 0); ca = int(ca or 0)
    vb = int(vb or 0); cb = int(cb or 0)
    conf = conf or 95

    pa_str = f"{ca/va*100:.2f}%" if va > 0 else "—"
    pb_str = f"{cb/vb*100:.2f}%" if vb > 0 else "—"

    rate_a = html.Span([pa_str, html.Span(" conv. rate", style={"fontSize": "12px", "color": MUTED})])
    rate_b = html.Span([pb_str, html.Span(" conv. rate", style={"fontSize": "12px", "color": MUTED})])

    # ── sample size ──
    ss = sample_size(ss_base or 4, ss_lift or 20)
    ss_display = f"{ss:,}" if ss else "—"

    # ── stats ──
    res = compute_stats(va, ca, vb, cb, conf) if va > 0 and vb > 0 else None

    if res is None:
        empty_fig = go.Figure(layout=dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        ))
        return rate_a, rate_b, html.Div(), [], empty_fig, empty_fig, [], ss_display

    # ── verdict banner ──
    VERDICTS = {
        "ship_b":   (GREEN,  "🚀  Ship Version B",
                     f"Deploy Version B. Lift is statistically significant at {conf}% confidence."),
        "keep_a":   (RED,    "⚠️  Keep Version A",
                     "Version B underperforms. Do not deploy."),
        "too_early":(YELLOW, "⏳  Too Early to Call",
                     "Not enough data yet. Keep collecting traffic."),
        "no_winner":(MUTED,  "🔍  No Significant Difference",
                     f"Lift is not significant at {conf}% confidence. Consider running longer."),
    }
    color, title, subtitle = VERDICTS[res["verdict"]]
    banner = html.Div([
        html.H2(title,    style={"color": color, "fontSize": "20px", "fontFamily": FONT_DISPLAY,
                                  "fontWeight": "800", "margin": "0 0 4px"}),
        html.P(subtitle,  style={"color": MUTED, "fontSize": "13px", "margin": "0"}),
    ], style={
        "background": f"{color}12", "border": f"1px solid {color}40",
        "borderRadius": "10px", "padding": "20px 24px",
    })

    # ── stat chips ──
    p_fmt = "< 0.0001" if res["p_value"] < 0.0001 else f"{res['p_value']:.4f}"
    chips = [
        stat_chip("Lift (relative)",  f"+{res['lift_rel']:.2f}%"    if res["lift_rel"] >= 0 else f"{res['lift_rel']:.2f}%",
                  GREEN if res["lift_rel"] >= 0 else RED),
        stat_chip("Abs. diff",        f"{res['lift_abs']:+.2f}pp",
                  GREEN if res["lift_abs"] >= 0 else RED),
        stat_chip("p-value",          p_fmt, GREEN if res["significant"] else YELLOW),
        stat_chip("z-score",          f"{res['z']:.3f}",            ACCENT),
    ]

    # ── distribution chart ──
    z_range = np.linspace(-5, 5, 400)
    y_norm  = stats.norm.pdf(z_range)
    z_crit  = res["z_crit"]

    dist_fig = go.Figure()

    # rejection regions
    dist_fig.add_trace(go.Scatter(
        x=z_range[z_range <= -z_crit], y=y_norm[z_range <= -z_crit],
        fill="tozeroy", fillcolor=f"{RED}30", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    dist_fig.add_trace(go.Scatter(
        x=z_range[z_range >= z_crit], y=y_norm[z_range >= z_crit],
        fill="tozeroy", fillcolor=f"{RED}30", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # normal curve
    dist_fig.add_trace(go.Scatter(
        x=z_range, y=y_norm,
        line=dict(color=BORDER, width=1.5),
        fill="tozeroy", fillcolor=f"{ACCENT}10",
        showlegend=False, hoverinfo="skip",
    ))
    # z-score line
    dist_fig.add_vline(x=res["z"], line_color=GREEN if res["significant"] else YELLOW,
                       line_width=2, line_dash="solid",
                       annotation_text=f"z = {res['z']:.3f}",
                       annotation_font_color=TEXT, annotation_font_size=10)
    # critical lines
    for xv in [-z_crit, z_crit]:
        dist_fig.add_vline(x=xv, line_color=f"{RED}80", line_width=1, line_dash="dot")

    dist_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=10), zeroline=False,
                   title=dict(text="z-score", font=dict(color=MUTED, size=10))),
        yaxis=dict(visible=False),
        hovermode=False,
    )

    # ── rate comparison chart ──
    rate_fig = go.Figure()
    rate_fig.add_trace(go.Bar(
        x=["Variant A (Control)", "Variant B (Challenger)"],
        y=[res["pa"] * 100, res["pb"] * 100],
        marker_color=[ACCENT, GREEN if res["pb"] >= res["pa"] else RED],
        text=[f"{res['pa']*100:.2f}%", f"{res['pb']*100:.2f}%"],
        textposition="outside",
        textfont=dict(color=TEXT, size=11, family=FONT_MONO),
        width=0.4,
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    rate_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(showgrid=True, gridcolor=f"{BORDER}66",
                   tickfont=dict(color=MUTED, size=10), ticksuffix="%"),
        showlegend=False,
    )

    # ── statistical detail ──
    sig_label = html.Span(
        "Significant" if res["significant"] else "Not Significant",
        style={
            "color": GREEN if res["significant"] else YELLOW,
            "background": f"{GREEN if res['significant'] else YELLOW}18",
            "padding": "2px 10px", "borderRadius": "4px",
            "fontSize": "11px", "fontWeight": "600",
        }
    )
    ci_lo_pp = res["ci_low"] * 100
    ci_hi_pp = res["ci_high"] * 100

    detail = html.Div([
        sig_label,
        html.Div(style={"marginTop": "12px"}, children=[
            detail_row("Conversion rate A",      f"{res['pa']*100:.2f}%"),
            detail_row("Conversion rate B",      f"{res['pb']*100:.2f}%"),
            detail_row(f"{conf}% CI for B − A",  f"[{ci_lo_pp:.2f}pp, {ci_hi_pp:.2f}pp]"),
            detail_row("Test",                   "Two-proportion z-test (pooled)"),
            detail_row("Post-hoc power",         f"{res['power']:.1f}%"),
            detail_row("Winner",                 "Version B" if res["verdict"] == "ship_b"
                                                 else "Version A" if res["verdict"] == "keep_a"
                                                 else "Inconclusive"),
        ]),
        html.P(
            f"Interpretation: There is a {res['p_value']*100:.2f}% probability of observing this "
            f"difference (or more extreme) if A and B truly converted at the same rate. "
            f"This is {'below' if res['significant'] else 'above'} the α = {res['alpha']:.2f} "
            f"threshold, so we {'reject' if res['significant'] else 'fail to reject'} the null hypothesis.",
            style={"color": MUTED, "fontSize": "11px", "lineHeight": "1.6",
                   "marginTop": "16px", "fontStyle": "italic"},
        ),
    ])

    return rate_a, rate_b, banner, chips, dist_fig, rate_fig, detail, ss_display


if __name__ == "__main__":
    app.run(debug=True, port=8051)