"""
Streamlit dashboard for volatility forecasting pipeline.
"""
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import json

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Vol Forecaster",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ───────────────────────────────────────────────────────────
DARK_BG    = "#0d0f14"
PANEL_BG   = "#13161e"
BORDER     = "#1e2330"
ACCENT     = "#4fffb0"        # neon-green
ACCENT2    = "#7b6cff"        # violet
WARN       = "#ff6b6b"
TEXT_DIM   = "#6b7280"
TEXT       = "#e2e8f0"
FONT       = "'Inter', 'Roboto Mono', monospace"

_AXIS_STYLE = dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(color=TEXT_DIM, size=10))

# Shared theme — does NOT include xaxis/yaxis so per-chart overrides never collide
PLOTLY_THEME = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT, family=FONT, size=11),
    margin=dict(l=12, r=12, t=32, b=12),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
        font=dict(color=TEXT_DIM, size=10),
    ),
    hoverlabel=dict(
        bgcolor=PANEL_BG, bordercolor=BORDER,
        font=dict(color=TEXT, size=11),
    ),
)


def _apply_axis_style(fig):
    """Apply default axis styling to every x/y axis on a figure."""
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: {FONT};
    background-color: {DARK_BG};
    color: {TEXT};
  }}

  /* hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.2rem 2rem 2rem; max-width: 1600px; }}

  /* sidebar */
  section[data-testid="stSidebar"] {{
    background-color: {PANEL_BG};
    border-right: 1px solid {BORDER};
  }}

  /* cards */
  .card {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    height: 100%;
  }}
  .card-header {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 0.6rem;
  }}
  .metric-value {{
    font-size: 1.9rem;
    font-weight: 600;
    color: {TEXT};
    line-height: 1.1;
  }}
  .metric-delta-pos {{
    font-size: 0.75rem;
    color: {ACCENT};
    margin-top: 0.2rem;
  }}
  .metric-delta-neg {{
    font-size: 0.75rem;
    color: {WARN};
    margin-top: 0.2rem;
  }}
  .metric-label {{
    font-size: 0.72rem;
    color: {TEXT_DIM};
    margin-top: 0.1rem;
  }}

  /* section titles */
  .section-title {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }}

  /* badge */
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .badge-green  {{ background: rgba(79,255,176,0.12); color: {ACCENT}; border: 1px solid rgba(79,255,176,0.3); }}
  .badge-red    {{ background: rgba(255,107,107,0.12); color: {WARN};  border: 1px solid rgba(255,107,107,0.3); }}
  .badge-violet {{ background: rgba(123,108,255,0.12); color: {ACCENT2}; border: 1px solid rgba(123,108,255,0.3); }}
  .badge-dim    {{ background: rgba(107,114,128,0.12); color: {TEXT_DIM}; border: 1px solid {BORDER}; }}

  /* nav tab strip */
  .nav-strip {{
    display: flex;
    gap: 0.3rem;
    margin-bottom: 1.4rem;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.5rem;
  }}

  /* lob table */
  .lob-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  .lob-table th {{
    color: {TEXT_DIM}; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.3rem 0.5rem; text-align: right;
    border-bottom: 1px solid {BORDER};
  }}
  .lob-table th:first-child {{ text-align: left; }}
  .lob-table td {{
    padding: 0.22rem 0.5rem; text-align: right;
    border-bottom: 1px solid rgba(30,35,48,0.6);
    font-variant-numeric: tabular-nums;
  }}
  .lob-table td:first-child {{ text-align: left; }}
  .bid-price {{ color: {ACCENT}; font-weight: 500; }}
  .ask-price {{ color: {WARN};   font-weight: 500; }}
  .depth-bar-bid {{
    display: inline-block; height: 8px;
    background: rgba(79,255,176,0.25); border-radius: 2px;
    vertical-align: middle; margin-right: 4px;
  }}
  .depth-bar-ask {{
    display: inline-block; height: 8px;
    background: rgba(255,107,107,0.25); border-radius: 2px;
    vertical-align: middle; margin-right: 4px;
  }}

  /* model comparison table */
  .model-table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  .model-table th {{
    color: {TEXT_DIM}; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.35rem 0.7rem;
    border-bottom: 1px solid {BORDER};
    text-align: right;
  }}
  .model-table th:first-child {{ text-align: left; }}
  .model-table td {{
    padding: 0.4rem 0.7rem; text-align: right;
    border-bottom: 1px solid rgba(30,35,48,0.6);
    font-variant-numeric: tabular-nums;
  }}
  .model-table td:first-child {{ text-align: left; font-weight: 500; }}
  .model-table tr:last-child td {{ border-bottom: none; }}
  .model-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .best {{ color: {ACCENT}; font-weight: 600; }}
  .worst {{ color: {WARN}; }}

  /* info box */
  .info-box {{
    background: rgba(123,108,255,0.07);
    border: 1px solid rgba(123,108,255,0.2);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: {TEXT_DIM};
    margin-top: 0.5rem;
  }}

  /* streamlit overrides */
  .stSelectbox > div > div, .stNumberInput > div > div > input {{
    background-color: {PANEL_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
  }}
  .stButton > button {{
    background: {ACCENT};
    color: #0d0f14;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
  }}
  .stButton > button:hover {{
    background: #3de8a0;
    color: #0d0f14;
  }}
  div[data-testid="stFileUploader"] {{
    border-color: {BORDER} !important;
    background: {PANEL_BG} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ── Data / model loading ─────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    try:
        from config import DataConfig, FeatureConfig, ModelConfig
        return DataConfig(), FeatureConfig(), ModelConfig()
    except Exception:
        return None, None, None


@st.cache_resource
def load_trained_models():
    try:
        import pickle
        models_dir = Path("models")
        models = {}
        for f in models_dir.glob("*.pkl"):
            with open(f, "rb") as fh:
                models[f.stem] = pickle.load(fh)
        return models or None
    except Exception:
        return None


def load_latest_data():
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return None
    files = list(data_dir.glob("*.parquet"))
    if not files:
        return None
    return pl.read_parquet(max(files, key=lambda p: p.stat().st_mtime))


def load_results():
    p = Path("results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── HTML helpers ─────────────────────────────────────────────────────────────
def card(content: str, header: str = "") -> str:
    h = f'<div class="card-header">{header}</div>' if header else ""
    return f'<div class="card">{h}{content}</div>'


def metric_html(label: str, value: str, delta: str = "", delta_positive: bool = True) -> str:
    delta_class = "metric-delta-pos" if delta_positive else "metric-delta-neg"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
<div class="metric-value">{value}</div>
{delta_html}
<div class="metric-label">{label}</div>
"""


def badge(text: str, style: str = "dim") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


# ── Navigation ───────────────────────────────────────────────────────────────
NAV_PAGES = ["Overview", "Order Book", "Features", "Models", "Data Collection"]

def render_topbar():
    # Brand + nav as columns
    logo, *nav_cols, ts_col = st.columns([2] + [1] * len(NAV_PAGES) + [1.5])
    with logo:
        st.markdown(
            f'<span style="font-size:1.15rem;font-weight:700;letter-spacing:-0.02em;color:{TEXT};">'
            f'▲ VOL<span style="color:{ACCENT};">CAST</span></span>',
            unsafe_allow_html=True,
        )
    with ts_col:
        st.markdown(
            f'<div style="text-align:right;font-size:0.7rem;color:{TEXT_DIM};padding-top:0.3rem;">'
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True,
        )

    # Use session state for active page
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"

    for col, name in zip(nav_cols, NAV_PAGES):
        with col:
            if st.button(name, key=f"nav_{name}", use_container_width=True):
                st.session_state["page"] = name

    st.markdown(f'<div style="border-bottom:1px solid {BORDER};margin-bottom:1.2rem;"></div>',
                unsafe_allow_html=True)
    return st.session_state["page"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(df, results):
    # ── KPI strip ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Live Market Snapshot</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    panels = []

    if df is not None and len(df) > 0:
        mid   = df["mid_price"][-1] if "mid_price" in df.columns else None
        spd   = df["spread"].mean() if "spread" in df.columns else None
        n_obs = len(df)
        dur_s = (
            (df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9
            if "timestamp_ns" in df.columns else None
        )
        panels = [
            ("Mid Price", f"${mid:,.2f}" if mid else "—", "", True),
            ("Avg Spread", f"${spd:.5f}" if spd else "—", "", True),
            ("Snapshots", f"{n_obs:,}", "", True),
            ("Duration", f"{dur_s:.1f}s" if dur_s else "—", "", True),
        ]
    else:
        panels = [("Mid Price","—","",True),("Avg Spread","—","",True),
                  ("Snapshots","—","",True),("Duration","—","",True)]

    # Add best RMSE KPI
    if results:
        best_rmse = min(v for k, v in results.items() if "rmse" in k and isinstance(v, float))
        panels.append(("Best RMSE (RV)", f"{best_rmse:.2e}", "LightGBM", True))

    for col, (label, val, delta, pos) in zip([c1,c2,c3,c4,c5], panels):
        with col:
            st.markdown(
                card(metric_html(label, val, delta, pos), ""),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price + spread chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Price & Spread Dynamics</div>', unsafe_allow_html=True)

    if df is not None and "mid_price" in df.columns and "spread" in df.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.04)

        xs = list(range(len(df)))

        fig.add_trace(go.Scatter(
            x=xs, y=df["mid_price"].to_list(),
            name="Mid Price",
            line=dict(color=ACCENT, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(79,255,176,0.04)",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xs, y=df["spread"].to_list(),
            name="Spread",
            line=dict(color=ACCENT2, width=1.2),
            fill="tozeroy",
            fillcolor="rgba(123,108,255,0.08)",
        ), row=2, col=1)

        fig.update_layout(**PLOTLY_THEME, height=420, showlegend=True)
        _apply_axis_style(fig)
        price_min = df["mid_price"].min()
        price_max = df["mid_price"].max()
        fig.update_yaxes(title_text="Price (USD)", range=[price_min - 500, price_max + 500], row=1, col=1, )
        fig.update_yaxes(title_text="Spread", row=2, col=1)
        fig.update_xaxes(title_text="Snapshot #", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        _no_data_box("No price data available — collect or upload data first.")

    # ── Model results summary ────────────────────────────────────────────────
    if results:
        st.markdown('<div class="section-title">Model Performance Summary</div>',
                    unsafe_allow_html=True)
        _render_results_table(results)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ORDER BOOK
# ══════════════════════════════════════════════════════════════════════════════
def page_order_book(df):
    st.markdown('<div class="section-title">Limit Order Book — Latest Snapshot</div>',
                unsafe_allow_html=True)

    if df is None or "bid_p0" not in df.columns:
        _no_data_box("Order book columns not available. Collect or upload LOB data first.")
        return

    latest = df.tail(1).to_dict(as_series=False)
    LEVELS = 10

    bids, asks = [], []
    for i in range(LEVELS):
        bp = (latest.get(f"bid_p{i}") or [float("nan")])[0]
        bq = (latest.get(f"bid_q{i}") or [0])[0]
        ap = (latest.get(f"ask_p{i}") or [float("nan")])[0]
        aq = (latest.get(f"ask_q{i}") or [0])[0]
        if not np.isnan(bp) and bp > 0:
            bids.append((bp, bq))
        if not np.isnan(ap) and ap > 0:
            asks.append((ap, aq))

    mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
    spread_val = (asks[0][0] - bids[0][0]) if bids and asks else None

    # ── Spread header
    if mid:
        h1, h2, h3, _ = st.columns([1.2, 1.2, 1.2, 3])
        with h1:
            st.markdown(card(metric_html("Mid Price", f"${mid:,.2f}")), unsafe_allow_html=True)
        with h2:
            st.markdown(card(metric_html("Spread", f"${spread_val:.5f}" if spread_val else "—")),
                        unsafe_allow_html=True)
        with h3:
            rel_spread = spread_val / mid * 1e4 if spread_val and mid else None
            st.markdown(card(metric_html("Relative Spread", f"{rel_spread:.2f} bps" if rel_spread else "—")),
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── LOB table
    max_q = max(([q for _, q in bids] + [q for _, q in asks]) or [1])

    def depth_bar(q, side):
        w = max(4, int(q / max_q * 80))
        cls = "depth-bar-bid" if side == "bid" else "depth-bar-ask"
        return f'<span class="{cls}" style="width:{w}px;"></span>'

    bid_rows = "".join(
        f"<tr>"
        f"<td class='bid-price'>${p:,.2f}</td>"
        f"<td>{depth_bar(q,'bid')}{q:,.4f}</td>"
        f"<td style='color:{TEXT_DIM}'>{i+1}</td>"
        f"</tr>"
        for i, (p, q) in enumerate(bids)
    )
    ask_rows = "".join(
        f"<tr>"
        f"<td class='ask-price'>${p:,.2f}</td>"
        f"<td>{depth_bar(q,'ask')}{q:,.4f}</td>"
        f"<td style='color:{TEXT_DIM}'>{i+1}</td>"
        f"</tr>"
        for i, (p, q) in enumerate(asks)
    )

    tbl_header = "<tr><th>Price</th><th>Quantity</th><th>Level</th></tr>"
    col_bid, col_ask = st.columns(2)
    with col_bid:
        st.markdown(
            card(f"<table class='lob-table'>{tbl_header}{bid_rows}</table>", "Bids"),
            unsafe_allow_html=True,
        )
    with col_ask:
        st.markdown(
            card(f"<table class='lob-table'>{tbl_header}{ask_rows}</table>", "Asks"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Depth chart
    st.markdown('<div class="section-title">Cumulative Depth</div>', unsafe_allow_html=True)
    bid_prices = [p for p, _ in bids]
    bid_cumq   = list(np.cumsum([q for _, q in bids]))
    ask_prices = [p for p, _ in asks]
    ask_cumq   = list(np.cumsum([q for _, q in asks]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bid_prices, y=bid_cumq, name="Bid Depth",
        line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(79,255,176,0.1)",
        mode="lines+markers", marker=dict(size=5, color=ACCENT),
    ))
    fig.add_trace(go.Scatter(
        x=ask_prices, y=ask_cumq, name="Ask Depth",
        line=dict(color=WARN, width=2),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
        mode="lines+markers", marker=dict(size=5, color=WARN),
    ))
    if mid:
        fig.add_vline(x=mid, line=dict(color=TEXT_DIM, width=1, dash="dot"),
                      annotation_text="Mid", annotation_font_color=TEXT_DIM)
    fig.update_layout(**PLOTLY_THEME, height=320)
    _apply_axis_style(fig)
    fig.update_yaxes(title_text="Cumulative Qty")
    fig.update_xaxes(title_text="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def page_features(df):
    st.markdown('<div class="section-title">Feature Engineering</div>', unsafe_allow_html=True)

    if df is None:
        _no_data_box("No data loaded.")
        return

    try:
        from engineer import build_features, feature_cols, clean
        from config import FeatureConfig
        cfg = FeatureConfig()
        feat_df = build_features(
            df,
            rv_windows=cfg.rv_windows,
            ofi_window=cfg.ofi_window,
            har_lags=cfg.har_lags,
            horizons=cfg.horizons,
            ticks_per_second=cfg.ticks_per_second,
        )
        fcols   = feature_cols(feat_df, har_lags=cfg.har_lags)

        if not fcols:
            _no_data_box("No feature columns found after engineering.")
            return

        # ── Feature stats KPIs
        c_cols = st.columns(min(len(fcols), 5))
        for col, fc in zip(c_cols, fcols[:5]):
            if fc in feat_df.columns:
                mu  = feat_df[fc].mean()
                std = feat_df[fc].std()
                with col:
                    st.markdown(
                        card(metric_html(fc, f"{mu:.4f}", f"σ {std:.4f}")),
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature time series
        st.markdown('<div class="section-title">Feature Time Series</div>', unsafe_allow_html=True)
        selected = st.multiselect(
            "Select features", fcols,
            default=fcols[:3],
            key="feat_sel",
        )

        if selected:
            fig = go.Figure()
            palette = [ACCENT, ACCENT2, WARN, "#ffd166", "#06d6a0", "#118ab2"]
            for i, fc in enumerate(selected):
                if fc in feat_df.columns:
                    fig.add_trace(go.Scatter(
                        y=feat_df[fc].to_list(),
                        name=fc,
                        line=dict(color=palette[i % len(palette)], width=1.3),
                    ))
            fig.update_layout(**PLOTLY_THEME, height=350)
            _apply_axis_style(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation heatmap
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        numeric = [c for c in fcols if c in feat_df.columns][:15]
        if len(numeric) > 1:
            corr = feat_df.select(numeric).to_pandas().corr()
            fig = px.imshow(
                corr,
                color_continuous_scale=[[0, WARN], [0.5, PANEL_BG], [1, ACCENT]],
                color_continuous_midpoint=0,
                aspect="auto",
            )
            fig.update_layout(**PLOTLY_THEME, height=420,
                              coloraxis_colorbar=dict(tickfont=dict(color=TEXT_DIM, size=9)))
            _apply_axis_style(fig)
            fig.update_traces(
                hoverongaps=False,
                hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Feature engineering failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ══════════════════════════════════════════════════════════════════════════════
def _render_results_table(results: dict):
    rows = [
        ("HAR-RV",    results.get("har_rmse"),  results.get("har_mae"),  None,                   None,                    None),
        ("LightGBM",  results.get("lgbm_rmse"), results.get("lgbm_mae"), results.get("auroc"),   results.get("auprc"),    results.get("brier")),
        ("LSTM",      results.get("lstm_rmse"),  None,                   results.get("lstm_auroc"), results.get("lstm_auprc"), results.get("lstm_brier")),
        ("TCN",       results.get("tcn_rmse"),   None,                   results.get("tcn_auroc"),  results.get("tcn_auprc"),  results.get("tcn_brier")),
    ]

    def _fmt(v, fmt=".5f"):
        return f"{v:{fmt}}" if v is not None else "—"

    # Find best per column
    rmse_vals = [r[1] for r in rows if r[1]]
    auroc_vals = [r[3] for r in rows if r[3]]
    best_rmse  = min(rmse_vals)  if rmse_vals  else None
    best_auroc = max(auroc_vals) if auroc_vals else None

    def cell(v, best, lower_is_better=True, fmt=".5f"):
        if v is None:
            return f"<td style='color:{TEXT_DIM}'>—</td>"
        is_best = best is not None and (
            (lower_is_better and abs(v - best) < 1e-12) or
            (not lower_is_better and abs(v - best) < 1e-12)
        )
        cls = "best" if is_best else ""
        return f"<td class='{cls}'>{_fmt(v, fmt)}</td>"

    tbl = f"""
    <table class="model-table">
      <tr>
        <th>Model</th>
        <th>RMSE ↓</th>
        <th>MAE ↓</th>
        <th>AUROC ↑</th>
        <th>AUPRC ↑</th>
        <th>Brier ↓</th>
      </tr>
    """
    for name, rmse, mae, auroc, auprc, brier in rows:
        tbl += (
            f"<tr><td>{name}</td>"
            + cell(rmse,  best_rmse,  True)
            + cell(mae,   None,       True)
            + cell(auroc, best_auroc, False, ".4f")
            + cell(auprc, None,       False, ".4f")
            + cell(brier, None,       True,  ".4f")
            + "</tr>"
        )
    tbl += "</table>"
    st.markdown(card(tbl, "Evaluation Metrics — Test Set"), unsafe_allow_html=True)


def _render_multi_horizon(results: dict):
    per_h = results.get("per_horizon")
    if not per_h:
        _no_data_box("No per-horizon data. Re-run train.py with updated code.")
        return

    horizons = list(per_h.keys())

    # ── RMSE by horizon grouped bar
    st.markdown('<div class="section-title">RV Forecast RMSE by Horizon</div>',
                unsafe_allow_html=True)
    model_keys   = [("HAR-RV",    "har",     "#ffd166"),
                    ("GARCH",     "garch",   WARN),
                    ("LightGBM",  "lgbm_rv", ACCENT),
                    ("LSTM",      "lstm_rv",  ACCENT2),
                    ("TCN",       "tcn_rv",  "#06d6a0")]
    fig = go.Figure()
    for label, key, color in model_keys:
        vals = [per_h[h].get(key, {}).get("rmse") for h in horizons]
        if any(v is not None for v in vals):
            fig.add_trace(go.Bar(
                name=label, x=horizons,
                y=[v if v is not None else 0 for v in vals],
                marker_color=color,
                text=[f"{v:.2e}" if v else "" for v in vals],
                textposition="outside",
                textfont=dict(color=TEXT, size=9),
            ))
    fig.update_layout(**PLOTLY_THEME, barmode="group", height=340, bargap=0.15)
    _apply_axis_style(fig)
    fig.update_yaxes(title_text="RMSE")
    fig.update_xaxes(title_text="Forecast Horizon")
    st.plotly_chart(fig, use_container_width=True)

    # ── Shock AUROC by horizon
    st.markdown('<div class="section-title">Shock Detection AUROC by Horizon</div>',
                unsafe_allow_html=True)
    shock_keys = [("LightGBM", "lgbm_shock", ACCENT),
                  ("LSTM",     "lstm_shock",  ACCENT2),
                  ("TCN",      "tcn_shock",   "#06d6a0")]
    fig2 = go.Figure()
    for label, key, color in shock_keys:
        vals = [per_h[h].get(key, {}).get("auroc") for h in horizons]
        if any(v is not None for v in vals):
            fig2.add_trace(go.Bar(
                name=label, x=horizons,
                y=[v if v is not None else 0 for v in vals],
                marker_color=color,
                text=[f"{v:.4f}" if v else "" for v in vals],
                textposition="outside",
                textfont=dict(color=TEXT, size=9),
            ))
    fig2.add_hline(y=0.5, line=dict(color=TEXT_DIM, dash="dot", width=1),
                   annotation_text="Random", annotation_font_color=TEXT_DIM)
    fig2.update_layout(**PLOTLY_THEME, barmode="group", height=320, bargap=0.15)
    _apply_axis_style(fig2)
    fig2.update_yaxes(title_text="AUROC", range=[0, 1.05])
    fig2.update_xaxes(title_text="Forecast Horizon")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Detailed per-horizon table
    st.markdown('<div class="section-title">Full Per-Horizon Metrics</div>',
                unsafe_allow_html=True)
    tbl = """<table class="model-table">
      <tr><th>Horizon</th><th>Model</th><th>RMSE</th><th>MAE</th>
          <th>QLIKE</th><th>AUROC</th><th>AUPRC</th></tr>"""
    for h in horizons:
        ph = per_h[h]
        for label, rv_key, sh_key in [
            ("HAR-RV",   "har",      None),
            ("GARCH",    "garch",    None),
            ("LGBM",     "lgbm_rv",  "lgbm_shock"),
            ("LSTM",     "lstm_rv",  "lstm_shock"),
            ("TCN",      "tcn_rv",   "tcn_shock"),
        ]:
            rv  = ph.get(rv_key, {})
            sh  = ph.get(sh_key, {}) if sh_key else {}
            if not rv:
                continue
            tbl += (
                f"<tr><td>{h}</td><td>{label}</td>"
                f"<td>{rv.get('rmse', float('nan')):.5f}</td>"
                f"<td>{rv.get('mae',  float('nan')):.5f}</td>"
                f"<td>{rv.get('qlike',float('nan')):.4f}</td>"
                f"<td>{sh.get('auroc',float('nan')):.4f}</td>"
                f"<td>{sh.get('auprc',float('nan')):.4f}</td></tr>"
            )
    tbl += "</table>"
    st.markdown(card(tbl, ""), unsafe_allow_html=True)


def _render_regime_analysis(results: dict):
    regime = results.get("regime_eval")
    if not regime:
        _no_data_box("No regime data. Re-run train.py with updated code.")
        return

    st.markdown('<div class="section-title">LGBM RV Performance by Market Regime (5s horizon)</div>',
                unsafe_allow_html=True)

    col_vol, col_spd = st.columns(2)
    for col, key, title in [
        (col_vol, "vol_regime",    "Volatility Regime"),
        (col_spd, "spread_regime", "Spread Regime"),
    ]:
        data = regime.get(key, {})
        if not data:
            continue
        with col:
            rows = ""
            for regime_val, metrics in sorted(data.items()):
                rmse = metrics.get("rmse", float("nan"))
                mae  = metrics.get("mae",  float("nan"))
                rows += (
                    f"<tr><td>{regime_val}</td>"
                    f"<td>{rmse:.6f}</td>"
                    f"<td>{mae:.6f}</td></tr>"
                )
            tbl = (
                f'<table class="model-table">'
                f"<tr><th>Regime</th><th>RMSE</th><th>MAE</th></tr>"
                f"{rows}</table>"
            )
            st.markdown(card(tbl, title), unsafe_allow_html=True)

    # Regime bar comparison
    st.markdown("<br>", unsafe_allow_html=True)
    for key, title in [("vol_regime", "RMSE — Volatility Regime"),
                       ("spread_regime", "RMSE — Spread Regime")]:
        data = regime.get(key, {})
        if not data:
            continue
        labels = sorted(data.keys())
        vals   = [data[r].get("rmse", 0) for r in labels]
        colors = [WARN if r == "high" else ACCENT for r in labels]
        fig = go.Figure(go.Bar(
            x=labels, y=vals, marker_color=colors,
            text=[f"{v:.2e}" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT, size=10),
        ))
        fig.update_layout(**PLOTLY_THEME, height=280, title=title, bargap=0.4)
        _apply_axis_style(fig)
        fig.update_yaxes(title_text="RMSE")
        st.plotly_chart(fig, use_container_width=True)


def _render_feature_ablation(results: dict):
    ablation = results.get("feature_ablation")
    if not ablation:
        _no_data_box("No feature ablation data. Re-run train.py with updated code.")
        return

    baseline = ablation.get("_baseline", {})
    groups   = {k: v for k, v in ablation.items() if k != "_baseline"}

    if not groups:
        _no_data_box("No ablation groups found.")
        return

    st.markdown(
        f'<div class="section-title">Feature Ablation — RMSE Impact (primary horizon)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="info-box">Baseline RMSE (all features): '
        f'<strong>{baseline.get("rmse", float("nan")):.6f}</strong>. '
        f'Positive ΔRMSE means removing this group hurts performance.</div>',
        unsafe_allow_html=True,
    )

    labels = list(groups.keys())
    deltas = [groups[k].get("delta_rmse", 0) for k in labels]
    colors = [WARN if d > 0 else ACCENT for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{d:+.2e}" for d in deltas],
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
    ))
    fig.update_layout(
        **PLOTLY_THEME, height=max(280, 60 * len(labels)),
        title="ΔRMSE when feature group is dropped",
    )
    _apply_axis_style(fig)
    fig.update_xaxes(title_text="ΔRMSE vs Baseline (positive = removal hurts)")
    st.plotly_chart(fig, use_container_width=True)

    # Ablation table
    rows = ""
    for g, m in ablation.items():
        delta = m.get("delta_rmse")
        delta_str = f"{delta:+.6f}" if delta is not None else "—"
        style = f"color:{WARN}" if (delta is not None and delta > 0) else f"color:{ACCENT}"
        rows += (
            f"<tr><td>{'Baseline' if g == '_baseline' else g}</td>"
            f"<td>{m.get('rmse', float('nan')):.6f}</td>"
            f"<td style='{style}'>{delta_str}</td></tr>"
        )
    tbl = (
        '<table class="model-table">'
        "<tr><th>Group Dropped</th><th>RMSE</th><th>ΔRMSE</th></tr>"
        f"{rows}</table>"
    )
    st.markdown(card(tbl, ""), unsafe_allow_html=True)


def page_models(results):
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    if not results:
        _no_data_box("No results found. Run <code>python train.py ...</code> to generate results.json.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overall Comparison",
        "Multi-Horizon",
        "Regime Analysis",
        "Feature Ablation",
    ])

    with tab1:
        # ── Metric table
        _render_results_table(results)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── RMSE comparison bar chart
        st.markdown('<div class="section-title">Regression Performance — RMSE</div>',
                    unsafe_allow_html=True)
        rmse_data = {
            "HAR-RV":   results.get("har_rmse"),
            "LightGBM": results.get("lgbm_rmse"),
            "LSTM":     results.get("lstm_rmse"),
            "TCN":      results.get("tcn_rmse"),
        }
        rmse_data = {k: v for k, v in rmse_data.items() if v is not None}
        if rmse_data:
            best_k = min(rmse_data, key=rmse_data.get)
            colors = [ACCENT if k == best_k else ACCENT2 for k in rmse_data]
            fig = go.Figure(go.Bar(
                x=list(rmse_data.keys()), y=list(rmse_data.values()),
                marker_color=colors,
                text=[f"{v:.2e}" for v in rmse_data.values()],
                textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            fig.update_layout(**PLOTLY_THEME, height=300, bargap=0.35)
            _apply_axis_style(fig)
            fig.update_yaxes(title_text="RMSE")
            st.plotly_chart(fig, use_container_width=True)

        # ── Classification metrics
        st.markdown('<div class="section-title">Shock Detection — Classification</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        auroc_data = {k: v for k, v in {
            "LightGBM": results.get("auroc"),
            "LSTM":     results.get("lstm_auroc"),
            "TCN":      results.get("tcn_auroc"),
        }.items() if v is not None}
        auprc_data = {k: v for k, v in {
            "LightGBM": results.get("auprc"),
            "LSTM":     results.get("lstm_auprc"),
            "TCN":      results.get("tcn_auprc"),
        }.items() if v is not None}

        def bar_chart(data, title, yref=None):
            if not data:
                return None
            best_k = max(data, key=data.get)
            clrs = [ACCENT if k == best_k else ACCENT2 for k in data]
            f = go.Figure(go.Bar(
                x=list(data.keys()), y=list(data.values()),
                marker_color=clrs,
                text=[f"{v:.4f}" for v in data.values()],
                textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            if yref is not None:
                f.add_hline(y=yref, line=dict(color=TEXT_DIM, dash="dot", width=1),
                            annotation_text="Random", annotation_font_color=TEXT_DIM)
            f.update_layout(**PLOTLY_THEME, height=260, title=title, bargap=0.4)
            _apply_axis_style(f)
            f.update_yaxes(range=[0, 1.05])
            return f

        with col_a:
            f = bar_chart(auroc_data, "AUROC (Shock)", yref=0.5)
            if f:
                st.plotly_chart(f, use_container_width=True)
        with col_b:
            f = bar_chart(auprc_data, "AUPRC (Shock)")
            if f:
                st.plotly_chart(f, use_container_width=True)

        # ── Diebold-Mariano
        dm = results.get("diebold_mariano")
        if dm:
            st.markdown('<div class="section-title">Diebold-Mariano Tests</div>',
                        unsafe_allow_html=True)
            dm_rows = ""
            for pair, stats in dm.items():
                better = stats.get("interpretation", "")
                sig    = stats.get("p_value", 1) < 0.05
                b_html = badge("Significant", "green") if sig else badge("Not Significant", "dim")
                dm_rows += (
                    f"<tr><td>{pair.replace('_vs_', ' vs ')}</td>"
                    f"<td>{stats.get('statistic', 0):.3f}</td>"
                    f"<td>{stats.get('p_value', 1):.2e}</td>"
                    f"<td>{better}</td>"
                    f"<td>{b_html}</td></tr>"
                )
            tbl = f"""<table class="model-table">
              <tr><th>Comparison</th><th>DM Stat</th><th>p-value</th>
                  <th>Result</th><th>Sig?</th></tr>
              {dm_rows}</table>"""
            st.markdown(card(tbl, "Statistical Forecast Comparison"), unsafe_allow_html=True)

    with tab2:
        _render_multi_horizon(results)

    with tab3:
        _render_regime_analysis(results)

    with tab4:
        _render_feature_ablation(results)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def page_data_collection():
    st.markdown('<div class="section-title">Data Ingestion</div>', unsafe_allow_html=True)

    col_live, col_upload = st.columns(2)

    with col_live:
        st.markdown(
            card("""
<div class="card-header">Kraken L2 Orderbook — Live Collector</div>
"""),
            unsafe_allow_html=True,
        )
        pair        = st.selectbox("Trading Pair", ["XBTUSD", "ETHUSD", "DOGEUSD", "SOLUSD"])
        n_snapshots = st.number_input("Snapshots", 10, 50000, 200, 10)
        mode        = st.radio(
            "Collection Mode",
            ["WebSocket (real-time stream)", "REST (poll)"],
            horizontal=True,
        )

        if mode == "REST (poll)":
            poll_interval = st.number_input("Poll Interval (s)", 0.1, 10.0, 1.0, 0.1)
        else:
            timeout_s = st.number_input("Timeout (s)", 10, 600, 120, 10)

        if st.button("Collect Live Data"):
            with st.spinner("Streaming from Kraken…"):
                try:
                    if mode == "WebSocket (real-time stream)":
                        from kraken_feed import collect_kraken_ws_snapshots
                        df = collect_kraken_ws_snapshots(
                            pair=pair, n_snapshots=n_snapshots,
                            levels=10, timeout_s=timeout_s,
                        )
                    else:
                        from kraken_feed import collect_kraken_snapshots
                        df = collect_kraken_snapshots(
                            pair=pair, n_snapshots=n_snapshots,
                            levels=10, poll_interval_s=poll_interval,
                        )
                    out_dir  = Path("data/raw")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"kraken_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    df.write_parquet(out_file)
                    st.success(f"Collected {len(df):,} snapshots → `{out_file.name}`")
                except Exception as e:
                    st.error(f"Collection failed: {e}")

    with col_upload:
        st.markdown(
            card("""
<div class="card-header">Upload Local Data File</div>
"""),
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Drop a .parquet or .csv file", type=["parquet", "csv"])
        if uploaded:
            try:
                df = (pl.read_parquet(uploaded) if uploaded.name.endswith(".parquet")
                      else pl.read_csv(uploaded))
                out_dir  = Path("data/raw")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.write_parquet(out_file)
                st.success(f"Saved {len(df):,} rows → `{out_file.name}`")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # ── Existing data files
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Available Data Files</div>', unsafe_allow_html=True)
    data_dir = Path("data/raw")
    if data_dir.exists():
        files = sorted(data_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            rows = ""
            for fp in files[:20]:
                size_kb = fp.stat().st_size / 1024
                mtime   = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                rows += f"<tr><td>{fp.name}</td><td>{size_kb:.1f} KB</td><td>{mtime}</td></tr>"
            tbl = f"""<table class="model-table">
              <tr><th>File</th><th>Size</th><th>Modified</th></tr>{rows}</table>"""
            st.markdown(card(tbl, ""), unsafe_allow_html=True)
        else:
            _no_data_box("No parquet files in data/raw/")
    else:
        _no_data_box("data/raw/ directory does not exist yet.")

    # ── CLI reminder
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-box">'
        f'<strong>Training is done via CLI:</strong><br>'
        f'<code>python train.py --source parquet --path data/raw/&lt;file&gt;.parquet</code>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────
def _no_data_box(msg: str):
    st.markdown(
        f'<div class="info-box">{msg}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = render_topbar()
    df      = load_latest_data()
    results = load_results()

    if page == "Overview":
        page_overview(df, results)
    elif page == "Order Book":
        page_order_book(df)
    elif page == "Features":
        page_features(df)
    elif page == "Models":
        page_models(results)
    elif page == "Data Collection":
        page_data_collection()


if __name__ == "__main__":
    main()