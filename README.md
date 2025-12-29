# Universal Supply Chain Forecaster (Amazon Chronos)

A GenAI-powered demand planning dashboard that uses **Amazon Chronos-Bolt (Foundation Model)** to perform Zero-Shot forecasting on retail inventory.

[Link to Deployed app](https://huggingface.co/spaces/Karthix1/chronos-supply-chain-forecaster) 
### The Problem
Traditional supply chain models (ARIMA, XGBoost) suffer from the **Cold Start Problem**: they require massive historical data to train a new model for every single product.

### The Solution
I deployed **Amazon Chronos**, a Time-Series Foundation Model based on the T5 architecture. It treats time series data as a language (tokenization), allowing it to forecast new products (Zero-Shot) without retraining.

### Key Features
*   **Foundation Model Backend:** Uses `chronos-bolt-small` for sub-second inference.
*   **Probabilistic Forecasting:** Generates P10/P50/P90 confidence intervals for **Safety Stock planning**.
*   **Universal Input:** Accepts both "Wide" (M5 Retail) and "Long" (Standard) CSV formats.

### Results
*   **Baseline (XGBoost):** WAPE 0.71
*   **Chronos (Zero-Shot):** WAPE 0.69 (**Beat baseline without training**)
*   **Impact:** Reduced forecasting pipeline setup time from Days to Seconds.

### Tech Stack
*   **Model:** Amazon Chronos (PyTorch / HuggingFace)
*   **Frontend:** Streamlit

*   **Deployment:** Hugging Face Spaces (CPU Inference)
