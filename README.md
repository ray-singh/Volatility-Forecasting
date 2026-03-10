# Volatility-Forecasting

This project aims to build an end-to-end market microstructure forecasting pipeline using Level-2 order book data to predict short-horizon market dynamics, including volatility and liquidity shocks (sudden bid-ask spread expansions). The system will reconstruct the limit order book from raw exchange data and engineer a rich set of microstructure features (such as order-flow imbalance, queue imbalance, depth ratios, realized volatility, and trade intensity) to capture short-term supply–demand dynamics in financial markets.

The project will evaluate both classical econometric models and modern machine learning approaches. Baseline models will include volatility models such as ARMA-GARCH and HAR-RV, while machine learning methods such as gradient boosting (LightGBM) and LSTM neural networks will be used to capture nonlinear patterns in high-frequency data. 

The pipeline will be implemented in Python using tools such as Polars/PyArrow for large-scale data processing, PyTorch and LightGBM for modeling, and MLflow for experiment tracking. The final system will function as a reproducible research platform for high-frequency market analysis, combining scalable data engineering with machine learning to better understand and predict short-term market behavior.
