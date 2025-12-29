 ## GenAI Supply Chain Forecasting System | Python, PyTorch, Amazon Chronos, Docker
- Designed and deployed a Zero-Shot demand forecasting engine using Amazon’s Chronos-Bolt Foundation Model, treating time-series data as a language tokenization problem.
- Achieved a 1.9% reduction in forecast error (WAPE) compared to a production-grade XGBoost baseline on the Walmart M5 dataset without requiring model training.
- Solved the "Cold Start" problem for new product launches by leveraging pre-trained transformer patterns, eliminating the need for historical lag features.
- Containerized the inference pipeline using Docker and deployed a Streamlit dashboard to Hugging Face Spaces, providing supply chain managers with real-time P90 probabilistic confidence intervals for safety stock planning.
