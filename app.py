"""
Streamlit dashboard for volatility forecasting pipeline.

Provides real-time order book visualization, feature engineering metrics,
model inference, and backtesting results.
"""
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json

# Avoid running expensive operations on every rerun
@st.cache_resource
def load_models_and_config():
    """Load pre-trained models and configuration."""
    try:
        from config import DataConfig, FeatureConfig, ModelConfig, PipelineConfig
        return {
            "data": DataConfig(),
            "features": FeatureConfig(),
            "models": ModelConfig(),
        }
    except ImportError:
        st.error("Configuration module not found. Run training pipeline first.")
        return None


@st.cache_resource
def load_trained_models():
    """Load serialized trained models."""
    try:
        import pickle
        models_dir = Path("models")
        if not models_dir.exists():
            return None

        models = {}
        for model_file in models_dir.glob("*.pkl"):
            with open(model_file, "rb") as f:
                models[model_file.stem] = pickle.load(f)
        return models if models else None
    except Exception as e:
        st.warning(f"Could not load models: {e}")
        return None


def load_latest_data():
    """Load latest processed data and features."""
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return None

    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    return pl.read_parquet(latest_file)


def render_header():
    """Render dashboard header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 📊 Volatility Forecasting Dashboard")
        st.markdown("Real-time market microstructure analysis and volatility prediction")
    with col2:
        st.metric("Refresh", datetime.now().strftime("%H:%M:%S"))


def render_data_overview(df):
    """Render data overview section."""
    st.subheader("📈 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Snapshots", len(df))

    with col2:
        if "mid_price" in df.columns:
            st.metric("Latest Mid Price", f"${df['mid_price'][-1]:.2f}")

    with col3:
        if "spread" in df.columns:
            avg_spread = df["spread"].mean()
            st.metric("Avg Spread", f"${avg_spread:.6f}")

    with col4:
        if "timestamp_ns" in df.columns:
            timestamps = pl.col("timestamp_ns")
            duration_s = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9
            st.metric("Duration", f"{duration_s:.1f}s")


def render_lob_visualization(df):
    """Render live limit order book snapshot."""
    st.subheader("📖 Current Limit Order Book (Latest)")

    if "bid_p0" not in df.columns or "ask_p0" not in df.columns:
        st.info("Order book data not available")
        return

    # Get latest snapshot
    latest = df.tail(1).to_dict(as_series=False)

    # Extract bid/ask levels
    levels = 10
    bids = []
    asks = []

    for i in range(levels):
        bid_p = latest.get(f"bid_p{i}", [float("nan")])[0]
        bid_q = latest.get(f"bid_q{i}", [0])[0]
        ask_p = latest.get(f"ask_p{i}", [float("nan")])[0]
        ask_q = latest.get(f"ask_q{i}", [0])[0]

        if not np.isnan(bid_p) and bid_p > 0:
            bids.append({"Price": bid_p, "Quantity": bid_q, "Side": "Bid"})
        if not np.isnan(ask_p) and ask_p > 0:
            asks.append({"Price": ask_p, "Quantity": ask_q, "Side": "Ask"})

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Bids (Buy Orders)**")
        if bids:
            bids_df = pl.DataFrame(bids).sort("Price", descending=True)
            st.dataframe(bids_df, use_container_width=True, hide_index=True)

    with col2:
        st.write("**Asks (Sell Orders)**")
        if asks:
            asks_df = pl.DataFrame(asks).sort("Price")
            st.dataframe(asks_df, use_container_width=True, hide_index=True)


def render_price_chart(df):
    """Render mid-price and spread chart."""
    st.subheader("💹 Price & Spread Dynamics")

    if "mid_price" not in df.columns or "spread" not in df.columns:
        st.info("Price data not available")
        return

    # Create figure with secondary y-axis
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=df["mid_price"].to_list(),
        name="Mid Price",
        yaxis="y1",
        line=dict(color="#1f77b4", width=2),
    ))

    fig.add_trace(go.Scatter(
        y=df["spread"].to_list(),
        name="Spread",
        yaxis="y2",
        line=dict(color="#ff7f0e", width=2),
    ))

    fig.update_layout(
        title="Price and Spread over Time",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(title="Mid Price ($)", title_font=dict(color="#1f77b4")),
        yaxis2=dict(title="Spread ($)", title_font=dict(color="#ff7f0e"),
                    overlaying="y", side="right"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_analysis(df):
    """Render feature engineering analysis."""
    st.subheader("🔬 Feature Engineering Metrics")

    from engineer import build_features, feature_cols, clean
    from config import FeatureConfig

    try:
        # Build features
        cfg_features = FeatureConfig()
        feat_df = build_features(df, rv_windows=cfg_features.rv_windows,
                                  ofi_window=cfg_features.ofi_window)
        fcols = feature_cols(feat_df, har_lags=cfg_features.har_lags)

        if not fcols:
            st.warning("No features available")
            return

        # Display feature statistics
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Feature Availability**")
            feature_status = {col: "✓" for col in fcols}
            for col, status in feature_status.items():
                st.text(f"{status} {col}")

        with col2:
            st.write("**Feature Ranges**")
            for col in fcols[:5]:  # Show first 5
                if col in feat_df.columns:
                    min_val = feat_df[col].min()
                    max_val = feat_df[col].max()
                    st.text(f"{col}: [{min_val:.4f}, {max_val:.4f}]")

        # Feature heatmap (correlation)
        st.write("**Feature Correlations**")
        numeric_cols = [c for c in fcols if c in feat_df.columns]
        if numeric_cols and len(numeric_cols) > 1:
            # Sample if too many columns
            if len(numeric_cols) > 15:
                numeric_cols = numeric_cols[:15]

            corr_data = feat_df.select(numeric_cols).to_pandas().corr()
            fig = px.imshow(
                corr_data,
                color_continuous_scale="RdBu_r",
                zmid=0,
                title="Feature Correlation Matrix",
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Feature engineering failed: {e}")


def render_model_predictions(df):
    """Render model inference and predictions."""
    st.subheader("🤖 Model Predictions")

    try:
        from engineer import build_features, feature_cols, clean
        from config import FeatureConfig
        from models import LGBMDualModel, HARRVModel
        import pickle

        cfg_features = FeatureConfig()
        feat_df = build_features(df, rv_windows=cfg_features.rv_windows,
                                  ofi_window=cfg_features.ofi_window)
        fcols = feature_cols(feat_df, har_lags=cfg_features.har_lags)

        if len(feat_df) < 100:
            st.warning("Not enough data for predictions (need >= 100 snapshots)")
            return

        # Get latest features
        latest_features = feat_df.tail(10).to_pandas()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Volatility Prediction (RV)**")
            # Try to load and use LGBM model
            try:
                models_dir = Path("models")
                if (models_dir / "lgbm_model.pkl").exists():
                    with open(models_dir / "lgbm_model.pkl", "rb") as f:
                        lgbm = pickle.load(f)

                    X_latest = latest_features[fcols].values[-1:] if fcols else None
                    if X_latest is not None and X_latest.shape[1] > 0:
                        rv_pred = lgbm.predict_rv(X_latest)
                        st.metric("Predicted RV (next period)", f"{rv_pred[0]:.4f}")
                        st.caption("Realized volatility estimate")
            except:
                st.info("Model not trained yet")

        with col2:
            st.write("**Shock Detection (Binary)**")
            try:
                models_dir = Path("models")
                if (models_dir / "lgbm_model.pkl").exists():
                    with open(models_dir / "lgbm_model.pkl", "rb") as f:
                        lgbm = pickle.load(f)

                    X_latest = latest_features[fcols].values[-1:] if fcols else None
                    if X_latest is not None and X_latest.shape[1] > 0:
                        shock_prob = lgbm.predict_shock_proba(X_latest)[0]
                        shock_class = "⚠️ Shock Likely" if shock_prob[1] > 0.5 else "✓ Normal"
                        st.metric("Shock Probability", f"{shock_prob[1]:.2%}",
                                 delta=shock_class)
                        st.caption("Liquidity shock detection")
            except:
                st.info("Model not trained yet")

    except Exception as e:
        st.warning(f"Could not generate predictions: {e}")


def render_backtesting_results():
    """Render backtesting and evaluation metrics."""
    st.subheader("📊 Backtesting & Evaluation")

    try:
        results_file = Path("results.json")
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)

            col1, col2, col3 = st.columns(3)

            with col1:
                if "rmse" in results:
                    st.metric("RMSE (RV)", f"{results['rmse']:.4f}")

            with col2:
                if "auroc" in results:
                    st.metric("AUROC (Shock)", f"{results['auroc']:.3f}")

            with col3:
                if "auprc" in results:
                    st.metric("AUPRC (Shock)", f"{results['auprc']:.3f}")

            # Show all results
            if st.checkbox("Show detailed results"):
                st.json(results)
        else:
            st.info("Run training pipeline to generate backtesting results")

    except Exception as e:
        st.warning(f"Could not load results: {e}")


def render_data_collection():
    """Render data collection interface."""
    st.subheader("🔄 Data Collection")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Kraken L2 Order Book**")
        pair = st.selectbox("Trading Pair", ["XBTUSD", "ETHUSD", "DOGEUSD"], key="pair")
        n_snapshots = st.number_input("Number of Snapshots", min_value=10,
                                      max_value=5000, value=100, step=10)
        poll_interval = st.number_input("Poll Interval (s)", min_value=0.1,
                                       max_value=10.0, value=1.0, step=0.1)

        if st.button("📥 Collect Live Data", key="collect"):
            with st.spinner("Collecting from Kraken API..."):
                try:
                    from kraken_feed import collect_kraken_snapshots
                    df = collect_kraken_snapshots(
                        pair=pair,
                        n_snapshots=n_snapshots,
                        levels=10,
                        poll_interval_s=poll_interval
                    )

                    # Save to data/raw
                    output_dir = Path("data/raw")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"kraken_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    df.write_parquet(output_file)

                    st.success(f"✓ Collected {len(df)} snapshots → {output_file.name}")
                except Exception as e:
                    st.error(f"Collection failed: {e}")

    with col2:
        st.write("**Upload Local Data**")
        uploaded_file = st.file_uploader("Upload .parquet file",
                                        type=["parquet", "csv"],
                                        key="upload")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".parquet"):
                    df = pl.read_parquet(uploaded_file)
                else:
                    df = pl.read_csv(uploaded_file)

                output_dir = Path("data/raw")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.write_parquet(output_file)

                st.success(f"✓ Uploaded {len(df)} rows → {output_file.name}")
            except Exception as e:
                st.error(f"Upload failed: {e}")


def render_training_pipeline():
    """Render training pipeline interface."""
    st.subheader("🚀 Training Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Select Data Source**")
        source = st.selectbox("Data Source", ["parquet", "kraken"], key="source")

        if source == "parquet":
            data_dir = Path("data/raw")
            files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
            if files:
                selected_file = st.selectbox(
                    "Select File",
                    [f.name for f in files],
                    key="file_select"
                )
                data_path = str(data_dir / selected_file)
            else:
                st.warning("No parquet files found in data/raw")
                data_path = None
        else:
            pair = st.selectbox("Kraken Pair", ["XBTUSD", "ETHUSD"], key="pair_train")
            data_path = None

    with col2:
        st.write("**Training Options**")
        train_lgbm = st.checkbox("Train LightGBM", value=True)
        train_har = st.checkbox("Train HAR-RV", value=True)
        train_lstm = st.checkbox("Train LSTM", value=False)

    if st.button("🏋️ Start Training", key="train"):
        with st.spinner("Training models..."):
            try:
                from train import train
                from config import PipelineConfig

                cfg = PipelineConfig()

                # Handle data_path for parquet source
                if source == "parquet" and data_path:
                    cfg.data.source = "parquet"
                    results = train(cfg, data_path=data_path)
                else:
                    cfg.data.source = source
                    if source == "kraken":
                        cfg.data.symbol = pair
                    results = train(cfg)

                st.success("✓ Training completed!")

                # Show results
                if isinstance(results, dict):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{results.get('rmse', 0):.4f}")
                    with col2:
                        st.metric("AUROC", f"{results.get('auroc', 0):.3f}")
                    with col3:
                        st.metric("AUPRC", f"{results.get('auprc', 0):.3f}")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.write(str(e))


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Volatility Forecasting",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Dashboard",
        "Data Collection",
        "Training",
        "Settings"
    ])

    # Load data
    df = load_latest_data()

    if page == "Dashboard":
        render_header()

        if df is not None and len(df) > 0:
            render_data_overview(df)

            col1, col2 = st.columns([2, 1])
            with col1:
                render_price_chart(df)
            with col2:
                render_lob_visualization(df)

            col1, col2 = st.columns(2)
            with col1:
                render_feature_analysis(df)
            with col2:
                render_model_predictions(df)

            render_backtesting_results()
        else:
            st.info("📥 No data loaded. Go to Data Collection tab to get started.")

    elif page == "Data Collection":
        render_data_collection()

    elif page == "Training":
        render_training_pipeline()

    elif page == "Settings":
        st.subheader("⚙️ Settings")
        st.write("**Configuration Summary**")
        config = load_models_and_config()
        if config:
            st.json({
                "data": config["data"].__dict__,
                "features": config["features"].__dict__,
                "models": config["models"].__dict__,
            })

        st.write("**Project Structure**")
        st.code("""
data/raw/                 # Raw LOB snapshots (parquet)
models/                   # Serialized trained models
results.json              # Backtesting results
config.py                 # Configuration dataclasses
engineer.py               # Feature engineering
dataset.py                # Dataset splits
models.py                 # Model classes
train.py                  # Training pipeline
kraken_feed.py            # Kraken API integration
app.py                    # This dashboard
        """)


if __name__ == "__main__":
    main()
