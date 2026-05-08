"""
Streamlit dashboard for volatility forecasting pipeline.
"""
import io
import os
import pickle as _pickle
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import json
from src.kraken_feed import KrakenWSStream
from src.bybit_feed import BybitWSStream

# ── GCS helpers ──────────────────────────────────────────────────────────────
_GCS_BUCKET = os.environ.get("GCS_BUCKET", "")

def _gcs_client():
    from google.cloud import storage
    return storage.Client()

def _gcs_download_bytes(blob_name: str) -> bytes | None:
    """Download a GCS blob as bytes. Returns None if bucket not configured or blob missing."""
    if not _GCS_BUCKET:
        return None
    try:
        bucket = _gcs_client().bucket(_GCS_BUCKET)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except Exception:
        return None

def _gcs_list_blobs(prefix: str) -> list[str]:
    """List blob names under a GCS prefix. Returns [] if bucket not configured."""
    if not _GCS_BUCKET:
        return []
    try:
        return [b.name for b in _gcs_client().list_blobs(_GCS_BUCKET, prefix=prefix)]
    except Exception:
        return []


class _RemapUnpickler(_pickle.Unpickler):
    """Remap bare module names saved before the src/ restructure."""
    def find_class(self, module, name):
        if module in ("models", "deep_models", "config", "engineer", "dataset"):
            module = f"src.{module}"
        return super().find_class(module, name)

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Vol Forecaster",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ───────────────────────────────────────────────────────────
DARK_BG = "#0d0f14"
PANEL_BG  = "#13161e"
BORDER= "#1e2330"
ACCENT= "#4fffb0"        # neon-green
ACCENT2 = "#7b6cff"        # violet
WARN = "#ff6b6b"
TEXT_DIM  = "#6b7280"
TEXT = "#e2e8f0"
FONT = "'Inter', 'Roboto Mono', monospace"

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
        from src.config import DataConfig, FeatureConfig, ModelConfig
        return DataConfig(), FeatureConfig(), ModelConfig()
    except Exception:
        return None, None, None


@st.cache_resource
def load_trained_models():
    models = {}
    if _GCS_BUCKET:
        for blob_name in _gcs_list_blobs("models/"):
            if not blob_name.endswith(".pkl"):
                continue
            stem = blob_name[len("models/"):-len(".pkl")]
            data = _gcs_download_bytes(blob_name)
            if data is None:
                continue
            try:
                models[stem] = _RemapUnpickler(io.BytesIO(data)).load()
            except Exception:
                pass
    else:
        try:
            for f in Path("models").glob("*.pkl"):
                with open(f, "rb") as fh:
                    models[f.stem] = _RemapUnpickler(fh).load()
        except Exception:
            pass
    return models or None


def load_latest_data():
    if _GCS_BUCKET:
        blobs = sorted(_gcs_list_blobs("orderbook/"))
        parquet_blobs = [b for b in blobs if b.endswith(".parquet")]
        if parquet_blobs:
            data = _gcs_download_bytes(parquet_blobs[-1])
            if data is not None:
                df = pl.read_parquet(io.BytesIO(data))
                if "mid" in df.columns and "mid_price" not in df.columns:
                    df = df.rename({"mid": "mid_price"})
                return df
        return None

    # Local fallback
    csv_dir = Path("data/csv")
    if csv_dir.exists():
        csv_files = sorted(csv_dir.glob("order-book-*.csv"))
        if csv_files:
            df = pl.read_csv(csv_files[-1], try_parse_dates=True)
            if "mid" in df.columns and "mid_price" not in df.columns:
                df = df.rename({"mid": "mid_price"})
            return df
    candidates = []
    for pattern in ("data/raw/*.parquet", "data/orderbook/*.parquet"):
        candidates.extend(Path().glob(pattern))
    if candidates:
        return pl.read_parquet(max(candidates, key=lambda p: p.stat().st_mtime))
    return None


def load_results():
    if _GCS_BUCKET:
        data = _gcs_download_bytes("results.json")
        if data is not None:
            return json.loads(data)
        return None
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
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
<div class="metric-value">{value}</div>
{delta_html}
<div class="metric-label">{label}</div>
"""


def badge(text: str, style: str = "dim") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


# ── Navigation ───────────────────────────────────────────────────────────────
NAV_PAGES = ["Features", "Models", "Live"]

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
        st.session_state["page"] = "Live"

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
        mid  = df["mid_price"][-1] if "mid_price" in df.columns else None
        spd  = df["spread"].mean() if "spread" in df.columns else None
        n_obs = len(df)
        if "timestamp_ns" in df.columns:
            dur_s = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9
        elif "ts" in df.columns:
            ts_col = df["ts"]
            dur_s = float((ts_col.max() - ts_col.min()).total_seconds())
        else:
            dur_s = None
        panels = [
            ("Mid Price", f"${mid:,.2f}" if mid else "—", "", True),
            ("Avg Spread", f"${spd:.5f}" if spd else "—", "", True),
            ("Snapshots", f"{n_obs:,}", "", True),
            ("Duration", f"{dur_s:.1f}s" if dur_s else "—", "", True),
        ]
    else:
        panels = [("Mid Price","—","",True),("Avg Spread","—","",True),
                  ("Snapshots","—","",True),("Duration","—","",True)]

    # Add best RMSE KPI — pick best LGBM RMSE across horizons
    if results:
        per_h = results.get("per_horizon") or {}
        lgbm_rmses = [v["lgbm_rv"]["rmse"] for v in per_h.values()
                      if "lgbm_rv" in v and "rmse" in v["lgbm_rv"]]
        best_rmse = min(lgbm_rmses) if lgbm_rmses else results.get("lgbm_rmse") or results.get("rmse")
        if best_rmse:
            panels.append(("Best RMSE (LGBM)", f"{best_rmse:.2e}", "1s horizon", True))

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
    LEVELS = sum(1 for i in range(20) if f"bid_p{i}" in df.columns)

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
    bid_cumq  = list(np.cumsum([q for _, q in bids]))
    ask_prices = [p for p, _ in asks]
    ask_cumq  = list(np.cumsum([q for _, q in asks]))

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
        from src.engineer import build_features, feature_cols
        from src.config import FeatureConfig
        cfg = FeatureConfig()
        feat_df = build_features(
            df,
            rv_windows=cfg.rv_windows,
            ofi_window=cfg.ofi_window,
            har_lags=cfg.har_lags,
            horizons=cfg.horizons,
            ticks_per_second=cfg.ticks_per_second,
            lob_levels=5,
        )
        fcols  = feature_cols(feat_df, har_lags=cfg.har_lags, lob_levels=5)

        if not fcols:
            _no_data_box("No feature columns found after engineering.")
            return

        # ── Feature stats KPIs
        c_cols = st.columns(min(len(fcols), 5))
        for col, fc in zip(c_cols, fcols[:5]):
            if fc in feat_df.columns:
                mu = feat_df[fc].mean()
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
            fig.update_xaxes(title_text="Snapshot #")
            fig.update_yaxes(title_text="Feature Value")
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation heatmap
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        numeric = [c for c in fcols if c in feat_df.columns][:15]
        if len(numeric) > 1:
            corr = feat_df.select(numeric).to_pandas().corr()
            fig = px.imshow(
                corr,
                color_continuous_scale=[[0, "#2166ac"], [0.5, PANEL_BG], [1, "#d6604d"]],
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
    # Pull 5s horizon (primary) from per_horizon if available, else fall back to flat legacy keys
    ph5 = (results.get("per_horizon") or {}).get("5s", {})
    har_rmse = ph5.get("har",  {}).get("rmse")  or results.get("har_rmse")
    har_mae  = ph5.get("har",  {}).get("mae")   or results.get("har_mae")
    garch_rmse = ph5.get("garch", {}).get("rmse")
    garch_mae = ph5.get("garch", {}).get("mae")
    lgbm_rmse = ph5.get("lgbm_rv",    {}).get("rmse") or results.get("lgbm_rmse")
    lgbm_mae = ph5.get("lgbm_rv",    {}).get("mae")  or results.get("lgbm_mae")
    lgbm_auroc = ph5.get("lgbm_shock", {}).get("auroc") or results.get("auroc")
    lgbm_auprc = ph5.get("lgbm_shock", {}).get("auprc") or results.get("auprc")
    lgbm_brier = ph5.get("lgbm_shock", {}).get("brier") or results.get("brier")
    tcn_rmse = ph5.get("tcn_rv", {}).get("rmse") or results.get("tcn_rmse")
    tcn_mae  = ph5.get("tcn_rv", {}).get("mae")
    tcn_auroc = ph5.get("tcn_shock", {}).get("auroc") or results.get("tcn_auroc")
    tcn_auprc = ph5.get("tcn_shock", {}).get("auprc") or results.get("tcn_auprc")
    tcn_brier = ph5.get("tcn_shock", {}).get("brier") or results.get("tcn_brier")
    tr_rmse  = ph5.get("transformer_rv", {}).get("rmse") or results.get("transformer_rmse")
    tr_mae = ph5.get("transformer_rv", {}).get("mae")
    tr_auroc = ph5.get("transformer_shock", {}).get("auroc") or results.get("transformer_auroc")
    tr_auprc = ph5.get("transformer_shock", {}).get("auprc") or results.get("transformer_auprc")
    tr_brier = ph5.get("transformer_shock", {}).get("brier") or results.get("transformer_brier")
    rows = [
        ("HAR-RV",      har_rmse,  har_mae,  None,       None,       None),
        ("GARCH",       garch_rmse, garch_mae, None,      None,       None),
        ("LightGBM",    lgbm_rmse, lgbm_mae, lgbm_auroc, lgbm_auprc, lgbm_brier),
        ("TCN",         tcn_rmse,  tcn_mae,  tcn_auroc,  tcn_auprc,  tcn_brier),
        ("Transformer", tr_rmse,   tr_mae,   tr_auroc,   tr_auprc,   tr_brier),
    ]
    rows = [r for r in rows if any(v is not None for v in r[1:])]

    def _fmt(v, fmt=".5f"):
        return f"{v:{fmt}}" if v is not None else "—"

    # Find best per column
    rmse_vals = [r[1] for r in rows if r[1]]
    auroc_vals = [r[3] for r in rows if r[3]]
    best_rmse = min(rmse_vals)  if rmse_vals  else None
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
    model_keys  = [("HAR-RV",      "har",            "#ffd166"),
                    ("GARCH",       "garch",          WARN),
                    ("LightGBM",    "lgbm_rv",        ACCENT),
                    ("TCN",         "tcn_rv",         "#06d6a0"),
                    ("Transformer", "transformer_rv",  ACCENT2)]
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
    shock_keys = [("LightGBM",    "lgbm_shock",        ACCENT),
                  ("TCN",         "tcn_shock",         "#06d6a0"),
                  ("Transformer", "transformer_shock",  ACCENT2)]
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
            ("HAR-RV",      "har",             None),
            ("GARCH",       "garch",           None),
            ("LGBM",        "lgbm_rv",         "lgbm_shock"),
            ("TCN",         "tcn_rv",          "tcn_shock"),
            ("Transformer", "transformer_rv",  "transformer_shock"),
        ]:
            rv = ph.get(rv_key, {})
            sh = ph.get(sh_key, {}) if sh_key else {}
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
                mae = metrics.get("mae",  float("nan"))
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
        vals  = [data[r].get("rmse", 0) for r in labels]
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
    groups  = {k: v for k, v in ablation.items() if k != "_baseline"}

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


def _render_latency_profiling():
    import io as _io
    import time as _time
    import pickle as _pickle
    import torch as _torch

    _device = "mps" if _torch.backends.mps.is_available() else "cpu"

    class _SafeUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: _torch.load(
                    _io.BytesIO(b), map_location=_device, weights_only=False
                )
            if module in ("models", "deep_models", "config", "engineer", "dataset"):
                module = f"src.{module}"
            return super().find_class(module, name)

    def _load_model(src):
        """src: file path (Path) or raw bytes."""
        fh = _io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else open(src, "rb")
        try:
            model = _SafeUnpickler(fh).load()
        finally:
            if not isinstance(src, (bytes, bytearray)):
                fh.close()
        if hasattr(model, "device"):
            model.device = _torch.device(_device)
        if hasattr(model, "net") and model.net is not None:
            model.net.to(_device)
        return model

    st.markdown('<div class="section-title">Inference Latency Benchmark</div>',
                unsafe_allow_html=True)

    n_runs = st.slider("Warmup + timed runs per model", min_value=10, max_value=500,
                        value=100, step=10)

    if not st.button("Run Benchmark"):
        st.markdown(
            '<div class="info-box">Click <strong>Run Benchmark</strong> to time '
            'predict_rv() and predict_shock_proba() for each loaded model.</div>',
            unsafe_allow_html=True,
        )
        return

    # Build list of (stem, source) where source is bytes (GCS) or Path (local).
    model_sources: list[tuple[str, "bytes | Path"]] = []
    if _GCS_BUCKET:
        for blob_name in sorted(_gcs_list_blobs("models/")):
            if not blob_name.endswith(".pkl"):
                continue
            stem = blob_name[len("models/"):-len(".pkl")]
            data = _gcs_download_bytes(blob_name)
            if data is not None:
                model_sources.append((stem, data))
    if not model_sources:
        for f in sorted(Path("models").glob("*.pkl")):
            model_sources.append((f.stem, f))
    if not model_sources:
        st.warning("No .pkl files found in GCS or local models/")
        return

    results_rows = []
    warmup = max(3, n_runs // 10)
    rng = np.random.default_rng(42)

    progress = st.progress(0.0, text="Benchmarking…")
    for idx, (stem, src) in enumerate(model_sources):
        try:
            model = _load_model(src)
        except Exception as e:
            results_rows.append({"Model": stem, "RV p50 (ms)": "load error",
                                  "RV p99 (ms)": str(e), "Shock p50 (ms)": "—",
                                  "Shock p99 (ms)": "—", "Throughput (pred/s)": "—"})
            continue

        # LGBM: call booster_.predict directly (avoids sklearn force_all_finite compat error).
        # Deep: remap to CPU and run net forward directly.
        rv_model  = getattr(model, "rv_model",    None)
        sh_model  = getattr(model, "shock_model", None)
        rv_booster = getattr(rv_model, "booster_", None)
        sh_booster = getattr(sh_model, "booster_", None)
        net  = getattr(model,    "net",       None)
        scaler= getattr(model,    "scaler",    None)

        n_feat = getattr(scaler, "n_features_in_", None) \
                  or getattr(rv_model, "n_features_in_", None) or 54
        seq_len = getattr(model, "seq_len", 1)

        rv_times, sh_times = [], []

        har_predict = getattr(model, "predict", None) if (rv_booster is None and net is None) else None

        if rv_booster is not None and sh_booster is not None:
            # LGBM: bypass sklearn wrapper
            X_dummy = rng.standard_normal((1, n_feat)).astype(np.float64)
            for _ in range(warmup + n_runs):
                t0 = _time.perf_counter()
                rv_booster.predict(X_dummy)
                rv_times.append(_time.perf_counter() - t0)
            for _ in range(warmup + n_runs):
                t0 = _time.perf_counter()
                sh_booster.predict(X_dummy)
                sh_times.append(_time.perf_counter() - t0)

        elif net is not None:
            # Deep model: remap weights to CPU, time net forward pass
            net.to("cpu")
            net.eval()
            X_np = rng.standard_normal((seq_len, n_feat)).astype(np.float32)
            if scaler is not None:
                X_np = scaler.transform(X_np).astype(np.float32)
            X_t = _torch.from_numpy(X_np).unsqueeze(0)  # (1, seq_len, n_feat)
            with _torch.no_grad():
                for _ in range(warmup + n_runs):
                    t0 = _time.perf_counter()
                    net(X_t)
                    rv_times.append(_time.perf_counter() - t0)
            sh_times = rv_times  # single forward produces both heads; report same latency

        elif har_predict is not None:
            # HAR-RV: params shape is (n_features,); X must be (n_samples, n_features).
            params = getattr(model, "params", None)
            n_params = params.shape[0] if params is not None else 55
            X_dummy = rng.standard_normal((1, n_params))
            for _ in range(warmup + n_runs):
                t0 = _time.perf_counter()
                har_predict(X_dummy)
                rv_times.append(_time.perf_counter() - t0)
            sh_times = []  # HAR is regression-only, no shock head

        else:
            results_rows.append({
                "Model": stem, "RV p50 (ms)": "unsupported",
                "RV p99 (ms)": "—", "Shock p50 (ms)": "—",
                "Shock p99 (ms)": "—", "Throughput (pred/s)": "—",
            })
            progress.progress((idx + 1) / len(model_sources), text=f"Skipped {stem}")
            continue

        rv_timed = np.array(rv_times[warmup:]) * 1000  # ms
        sh_timed = np.array(sh_times[warmup:]) * 1000

        def _fmt(arr, pct):
            return f"{np.percentile(arr, pct):.3f}" if len(arr) else "—"

        # throughput = predictions per second (one full inference = rv + shock)
        combined_s = (rv_timed.sum() + sh_timed.sum()) / 1000
        tput = f"{n_runs / combined_s:,.0f}" if combined_s > 0 else "—"

        results_rows.append({
            "Model":               stem,
            "RV p50 (ms)":         _fmt(rv_timed, 50),
            "RV p99 (ms)":         _fmt(rv_timed, 99),
            "Shock p50 (ms)":      _fmt(sh_timed, 50) if len(sh_timed) else "N/A",
            "Shock p99 (ms)":      _fmt(sh_timed, 99) if len(sh_timed) else "N/A",
            "Throughput (pred/s)": tput,
        })
        progress.progress((idx + 1) / len(model_sources), text=f"Benchmarked {stem}")

    progress.empty()

    if not results_rows:
        st.warning("No models could be benchmarked.")
        return

    # ── Latency table
    header_cols = list(results_rows[0].keys())
    th = "".join(f"<th>{c}</th>" for c in header_cols)
    rows_html = ""
    for row in results_rows:
        rows_html += "<tr>" + "".join(f"<td>{row[c]}</td>" for c in header_cols) + "</tr>"
    tbl = f'<table class="model-table"><tr>{th}</tr>{rows_html}</table>'
    device_note = "LGBM: cpu · Deep models: cpu (benchmarked on CPU for isolation)"
    st.markdown(card(tbl, f"Results — {n_runs} timed runs, single-row input · {device_note}"),
                unsafe_allow_html=True)

    # ── p50 bar chart for RV latency
    try:
        chart_data = {r["Model"]: float(r["RV p50 (ms)"]) for r in results_rows
                      if r["RV p50 (ms)"] not in ("—", "load error")}
    except ValueError:
        chart_data = {}

    if chart_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">RV Inference p50 Latency</div>',
                    unsafe_allow_html=True)
        best_k = min(chart_data, key=chart_data.get)
        colors = [ACCENT if k == best_k else ACCENT2 for k in chart_data]
        fig = go.Figure(go.Bar(
            x=list(chart_data.keys()),
            y=list(chart_data.values()),
            marker_color=colors,
            text=[f"{v:.3f} ms" for v in chart_data.values()],
            textposition="outside",
            textfont=dict(color=TEXT, size=10),
        ))
        fig.update_layout(**PLOTLY_THEME, height=300, bargap=0.35)
        _apply_axis_style(fig)
        fig.update_yaxes(title_text="Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)


def page_models(results):
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    if not results:
        _no_data_box("No results found. Run <code>python train.py ...</code> to generate results.json.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Comparison",
        "Multi-Horizon",
        "Regime Analysis",
        "Feature Ablation",
        "Latency Profiling",
    ])

    with tab1:
        # ── Metric table
        _render_results_table(results)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── RMSE comparison bar chart
        st.markdown('<div class="section-title">Regression Performance — RMSE</div>',
                    unsafe_allow_html=True)
        ph5 = (results.get("per_horizon") or {}).get("5s", {})
        rmse_data = {
            "HAR-RV":      ph5.get("har",            {}).get("rmse") or results.get("har_rmse"),
            "GARCH":       ph5.get("garch",          {}).get("rmse"),
            "LightGBM":    ph5.get("lgbm_rv",        {}).get("rmse") or results.get("lgbm_rmse"),
            "TCN":         ph5.get("tcn_rv",         {}).get("rmse") or results.get("tcn_rmse"),
            "Transformer": ph5.get("transformer_rv", {}).get("rmse") or results.get("transformer_rmse"),
        }
        rmse_data = {k: v for k, v in rmse_data.items() if v is not None}
        _model_colors = {"HAR-RV": "#ffd166", "GARCH": WARN, "LightGBM": ACCENT,
                         "TCN": "#06d6a0", "Transformer": ACCENT2}
        if rmse_data:
            colors = [_model_colors.get(k, TEXT_DIM) for k in rmse_data]
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

        ph5_shock = (results.get("per_horizon") or {}).get("5s", {}).get("lgbm_shock", {})
        auroc_data = {k: v for k, v in {
            "LightGBM": ph5_shock.get("auroc") or results.get("auroc"),
        }.items() if v is not None}
        auprc_data = {k: v for k, v in {
            "LightGBM": ph5_shock.get("auprc") or results.get("auprc"),
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
                sig = stats.get("p_value", 1) < 0.05
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

    with tab5:
        _render_latency_profiling()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_lgbm_models() -> dict:
    """Load per-horizon LGBM models. Pulls from GCS when GCS_BUCKET is set, else local."""
    models = {}
    for h in ("1s", "5s", "30s"):
        if _GCS_BUCKET:
            data = _gcs_download_bytes(f"models/lgbm_model_{h}.pkl")
            if data is not None:
                try:
                    models[h] = _RemapUnpickler(io.BytesIO(data)).load()
                except Exception:
                    pass
        else:
            p = Path(f"models/lgbm_model_{h}.pkl")
            if p.exists():
                with open(p, "rb") as fh:
                    models[h] = _RemapUnpickler(fh).load()
    return models


def _build_live_features(rows: list[dict]) -> "tuple | None":
    """
    Run feature engineering on a list of raw LOB rows.
    Returns (feat_df, fcols) where feat_df retains all rows but NaN/inf cells
    are replaced with 0 so the last row is always valid for prediction.
    Returns None if feature engineering fails entirely.
    """
    try:
        from src.engineer import build_features, feature_cols
        from src.config import FeatureConfig
        cfg = FeatureConfig()
        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col("timestamp_ns").cast(pl.Int64))
        feat_df = build_features(
            df,
            rv_windows=cfg.rv_windows,
            ofi_window=cfg.ofi_window,
            har_lags=cfg.har_lags,
            horizons=cfg.horizons,
            ticks_per_second=cfg.ticks_per_second,
            lob_levels=5,
        )
        fcols = feature_cols(feat_df, har_lags=cfg.har_lags, lob_levels=5)
        if not fcols:
            return None
        # Replace null/NaN/inf with 0 for live inference — don't drop rows.
        # Rolling windows produce Polars nulls (shift/rolling_*) and IEEE NaN
        # (sqrt of near-zero sum); fill both so the last row is always valid.
        float_fcols = [c for c in fcols if feat_df[c].dtype in (pl.Float32, pl.Float64)]
        feat_df = feat_df.with_columns([
            pl.when(pl.col(c).is_null() | pl.col(c).is_nan() | pl.col(c).is_infinite())
              .then(pl.lit(0.0))
              .otherwise(pl.col(c))
              .alias(c)
            for c in float_fcols
        ])
        return feat_df, fcols
    except Exception as e:
        st.warning(f"Feature engineering failed: {e}")
        return None


_API_URL = os.environ.get("API_URL", "").rstrip("/")


def _predict_via_api(rows: list[dict]) -> dict | None:
    """
    Call the FastAPI /predict endpoint with raw LOB rows.
    Returns preds dict on success, None on any failure.
    """
    try:
        import requests as _requests
        resp = _requests.post(
            f"{_API_URL}/predict",
            json={"snapshots": rows},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            h: {"rv": v["rv"], "shock_p": v["shock_probability"]}
            for h, v in data.get("predictions", {}).items()
        }
    except Exception:
        return None


def _predict_live(feat_df, fcols: list[str], lgbm_models: dict) -> dict:
    """
    Run LGBM predictions on the last valid row of feat_df.
    Returns dict with rv predictions and shock probabilities per horizon.
    """
    preds = {}
    if feat_df is None or len(feat_df) == 0:
        return preds
    last_row = feat_df.tail(1).select(fcols).to_numpy()
    for h, model in lgbm_models.items():
        try:
            n_feats = model.rv_model.n_features_in_
            X = last_row[:, :n_feats]
            rv_pred = float(model.predict_rv(X)[0])
            shock_proba = model.predict_shock_proba(X)[0]
            shock_p = float(shock_proba[1]) if len(shock_proba) > 1 else float(shock_proba[0])
            preds[h] = {"rv": rv_pred, "shock_p": shock_p}
        except Exception as e:
            st.warning(f"[debug] predict {h} failed: {e}")
    return preds


def page_live():
    st.markdown('<div class="section-title">Live Order Book + Predictions</div>',
                unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctl, col_status = st.columns([3, 2])
    with col_ctl:
        exchange = st.selectbox("Exchange", ["Bybit (inverse)", "Kraken (spot)"],
                                key="live_exchange")
        is_bybit = exchange.startswith("Bybit")
        pair = st.selectbox(
            "Pair",
            ["BTCUSD", "ETHUSD", "SOLUSD"] if is_bybit else ["XBTUSD", "ETHUSD"],
            key="live_pair",
        )
        if not is_bybit:
            st.warning(
                "**Domain mismatch:** Models were trained on Bybit inverse-perpetual data. "
                "Kraken spot microstructure (different spread regime, no funding rate) "
                "is out-of-distribution — predictions may be unreliable.",
                icon="⚠️",
            )

    def _make_stream(is_bybit, pair):
        if is_bybit:
            return BybitWSStream(symbol=pair, levels=5, maxlen=600)
        return KrakenWSStream(pair=pair, levels=5, maxlen=600)

    # ── Stream lifecycle ──────────────────────────────────────────────────────
    stream_key = f"live_stream_{exchange}_{pair}"
    if stream_key not in st.session_state:
        st.session_state[stream_key] = None

    # Auto-start stream on first visit
    if st.session_state[stream_key] is None or not st.session_state[stream_key].is_running:
        stream = _make_stream(is_bybit, pair)
        stream.start()
        st.session_state[stream_key] = stream

    c_start, c_stop = col_ctl.columns(2)
    with c_start:
        if st.button("▶ Restart Stream", key="live_start"):
            s = st.session_state.get(stream_key)
            if s is not None:
                s.stop()
            stream = _make_stream(is_bybit, pair)
            stream.start()
            st.session_state[stream_key] = stream

    with c_stop:
        if st.button("■ Stop", key="live_stop"):
            s = st.session_state.get(stream_key)
            if s is not None:
                s.stop()
                st.session_state[stream_key] = None

    stream = st.session_state.get(stream_key)
    exch_label = "Bybit" if is_bybit else "Kraken"
    with col_status:
        if stream and stream.is_running:
            n_rows = len(stream.snapshot_deque)
            persisted = getattr(stream, "rows_persisted", 0)
            st.markdown(
                f'<div class="info-box" style="margin-top:1.6rem;">'
                f'<span style="color:{ACCENT}">● LIVE</span> &nbsp;·&nbsp; '
                f'{exch_label} {pair} &nbsp;·&nbsp; '
                f'{n_rows} in memory &nbsp;·&nbsp; '
                f'<span style="color:{TEXT_DIM}">{persisted:,} saved to disk</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-box" style="margin-top:1.6rem;">'
                f'<span style="color:{TEXT_DIM}">○ Stopped</span> &nbsp;·&nbsp; '
                f'Press ▶ to connect</div>',
                unsafe_allow_html=True,
            )

    n_snapshots = len(stream.snapshot_deque) if stream else 0
    # rv_300 needs 300 ticks; require 310 so the last row has a full window.
    _MIN_ROWS = 310
    if stream is None or not stream.is_running or n_snapshots < _MIN_ROWS:
        pct = int(n_snapshots / _MIN_ROWS * 100) if stream else 0
        _no_data_box(
            f"Warming up… {n_snapshots}/{_MIN_ROWS} snapshots "
            f"({pct}%) — predictions appear once the 300-tick RV window fills."
        )
        if stream and stream.is_running:
            st.rerun()
        return

    rows = list(stream.snapshot_deque)
    latest = rows[-1]

    # ── KPI strip: mid, spread, predictions ──────────────────────────────────
    lgbm_models = _load_lgbm_models()
    result = _build_live_features(rows)
    feat_df, fcols = result if result is not None else (None, [])
    preds = {}
    if _API_URL:
        api_preds = _predict_via_api(rows)
        if api_preds is not None:
            preds = api_preds
    if not preds and feat_df is not None:
        preds = _predict_live(feat_df, fcols, lgbm_models)

    mid = latest.get("mid_price", 0)
    spread = latest.get("spread", 0)

    kpi_cols = st.columns(2 + len(preds) * 2)
    with kpi_cols[0]:
        st.markdown(card(metric_html("Mid Price", f"${mid:,.2f}")), unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(card(metric_html("Spread", f"${spread:.5f}")), unsafe_allow_html=True)

    col_idx = 2
    for h in ("1s", "5s", "30s"):
        if h not in preds:
            continue
        rv = preds[h]["rv"]
        shock = preds[h]["shock_p"]
        shock_color = WARN if shock > 0.5 else ACCENT
        shock_label = "HIGH SHOCK" if shock > 0.5 else f"{shock:.1%}"
        with kpi_cols[col_idx]:
            st.markdown(
                card(metric_html(f"RV Forecast ({h})", f"{rv:.2e}")),
                unsafe_allow_html=True,
            )
        with kpi_cols[col_idx + 1]:
            st.markdown(
                card(f'<div class="metric-value" style="color:{shock_color}">{shock_label}</div>'
                     f'<div class="metric-label">Shock prob ({h})</div>'),
                unsafe_allow_html=True,
            )
        col_idx += 2

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live LOB table ────────────────────────────────────────────────────────
    LEVELS = sum(1 for i in range(10) if f"bid_p{i}" in latest)
    bids, asks = [], []
    for i in range(LEVELS):
        bp = latest.get(f"bid_p{i}", float("nan"))
        bq = latest.get(f"bid_q{i}", 0.0)
        ap = latest.get(f"ask_p{i}", float("nan"))
        aq = latest.get(f"ask_q{i}", 0.0)
        if bp and not (bp != bp):   # nan check
            bids.append((bp, bq))
        if ap and not (ap != ap):
            asks.append((ap, aq))

    max_q = max(([q for _, q in bids] + [q for _, q in asks]) or [1])

    def depth_bar(q, side):
        w = max(4, int(q / max_q * 80))
        cls = "depth-bar-bid" if side == "bid" else "depth-bar-ask"
        return f'<span class="{cls}" style="width:{w}px;"></span>'

    tbl_header = "<tr><th>Price</th><th>Quantity</th><th>Level</th></tr>"
    bid_rows = "".join(
        f"<tr><td class='bid-price'>${p:,.2f}</td>"
        f"<td>{depth_bar(q,'bid')}{q:,.4f}</td>"
        f"<td style='color:{TEXT_DIM}'>{i+1}</td></tr>"
        for i, (p, q) in enumerate(bids)
    )
    ask_rows = "".join(
        f"<tr><td class='ask-price'>${p:,.2f}</td>"
        f"<td>{depth_bar(q,'ask')}{q:,.4f}</td>"
        f"<td style='color:{TEXT_DIM}'>{i+1}</td></tr>"
        for i, (p, q) in enumerate(asks)
    )
    col_bid, col_ask = st.columns(2)
    with col_bid:
        st.markdown(card(f"<table class='lob-table'>{tbl_header}{bid_rows}</table>", "Bids"),
                    unsafe_allow_html=True)
    with col_ask:
        st.markdown(card(f"<table class='lob-table'>{tbl_header}{ask_rows}</table>", "Asks"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rolling mid-price + spread chart ─────────────────────────────────────
    st.markdown('<div class="section-title">Live Price & Spread</div>', unsafe_allow_html=True)
    n_chart = min(len(rows), 300)
    chart_rows = rows[-n_chart:]
    mids = [r["mid_price"] for r in chart_rows]
    spreads = [r["spread"]    for r in chart_rows]
    xs = list(range(len(chart_rows)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=xs, y=mids, name="Mid Price",
                             line=dict(color=ACCENT, width=1.5),
                             fill="tozeroy", fillcolor="rgba(79,255,176,0.04)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=spreads, name="Spread",
                             line=dict(color=ACCENT2, width=1.2),
                             fill="tozeroy", fillcolor="rgba(123,108,255,0.08)"),
                  row=2, col=1)
    price_min = min(mids)
    price_max = max(mids)
    fig.update_layout(**PLOTLY_THEME, height=380, showlegend=True)
    _apply_axis_style(fig)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     range=[price_min - 10, price_max + 10])
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_xaxes(title_text="Snapshot #", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling RV forecast chart ─────────────────────────────────────────────
    if result is not None and len(feat_df) > 1:
        st.markdown('<div class="section-title">Rolling Volatility Forecast</div>',
                    unsafe_allow_html=True)
        rv_fig = go.Figure()
        palette = {"1s": ACCENT, "5s": ACCENT2, "30s": WARN}
        for h, model in lgbm_models.items():
            try:
                n_feats = model.rv_model.n_features_in_
                X = feat_df.select(fcols).to_numpy()[:, :n_feats]
                rv_series = model.predict_rv(X)
                rv_fig.add_trace(go.Scatter(
                    y=rv_series,
                    name=f"RV {h}",
                    line=dict(color=palette.get(h, TEXT), width=1.3),
                ))
            except Exception:
                pass
        rv_fig.update_layout(**PLOTLY_THEME, height=280)
        _apply_axis_style(rv_fig)
        rv_fig.update_yaxes(title_text="Predicted RV")
        rv_fig.update_xaxes(title_text="Snapshot #")
        st.plotly_chart(rv_fig, use_container_width=True)

    # Poll until the WebSocket pushes a new snapshot, then rerun
    import time as _time
    live_stream = st.session_state.get(stream_key)
    if live_stream and live_stream.is_running:
        n_before = len(live_stream.snapshot_deque)
        for _ in range(20):  # wait up to 1s in 50ms ticks
            _time.sleep(0.05)
            if len(live_stream.snapshot_deque) != n_before:
                break
        st.rerun()


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
    df = load_latest_data()
    results = load_results()

    if page == "Features":
        page_features(df)
    elif page == "Models":
        page_models(results)
    elif page == "Live":
        page_live()


if __name__ == "__main__":
    main()