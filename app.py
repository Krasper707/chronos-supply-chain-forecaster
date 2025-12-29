import streamlit as st
import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Chronos Supply Chain Planner", layout="wide", page_icon="📦")
st.title("📦 Universal Supply Chain Forecaster")
st.markdown("""
**Powered by Amazon Chronos-Bolt.**  
Supports both **Standard Time Series** (Long) and **M5/Retail Matrices** (Wide).
""")

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    device = "cpu" 
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=device,
        torch_dtype=torch.float32,
    )
    return pipeline

with st.spinner("Initializing AI Brain..."):
    pipeline = load_model()

# ==========================================
# 3. SIDEBAR - DATA CONFIG
# ==========================================
st.sidebar.header("1. Data Setup")
data_format = st.sidebar.radio("Data Format:", ["Long Format (Standard)", "Wide Format (M5 Style)"])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

df = None
series_data = None  # This will hold the final 1D array for forecasting
final_date_index = None # This will hold the dates for plotting

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(raw_df)} rows.")

    # ==========================================
    # LOGIC FOR WIDE FORMAT (M5 Style)
    # ==========================================
    if data_format == "Wide Format (M5 Style)":
        st.sidebar.subheader("Wide Format Mapping")
        
        # 1. ID Selection
        id_col = st.sidebar.selectbox("ID Column (e.g., item_id)", raw_df.columns)
        
        # 2. Select Specific Item
        selected_id = st.sidebar.selectbox("Select Item to Forecast", raw_df[id_col].unique())
        
        # 3. Extract the Series
        # We assume all OTHER columns that look like numbers/days are the time series
        # For M5, columns are 'd_1', 'd_2'...
        row = raw_df[raw_df[id_col] == selected_id].iloc[0]
        
        # Filter strictly for columns starting with 'd_' if it's M5, or just take numeric cols
        if 'd_1' in raw_df.columns:
            # Smart detection for M5
            day_cols = [c for c in raw_df.columns if c.startswith('d_')]
            series_values = row[day_cols].values.astype(float)
        else:
            # Fallback: take all numeric columns except ID
            numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
            if id_col in numeric_cols: numeric_cols.remove(id_col)
            series_values = row[numeric_cols].values.astype(float)

        series_data = series_values
        
        # Create Dummy Dates (since Wide format usually lacks actual dates)
        start_date = st.sidebar.date_input("Start Date (for plotting)", pd.to_datetime("2011-01-29"))
        final_date_index = pd.date_range(start=start_date, periods=len(series_data), freq='D')

    # ==========================================
    # LOGIC FOR LONG FORMAT (Standard)
    # ==========================================
    else:
        st.sidebar.subheader("Long Format Mapping")
        date_col = st.sidebar.selectbox("Date Column", raw_df.columns)
        target_col = st.sidebar.selectbox("Value Column", raw_df.columns)
        
        # Optional ID Filter
        if st.sidebar.checkbox("Filter by ID?"):
            id_col_long = st.sidebar.selectbox("ID Column", raw_df.columns)
            selected_id_long = st.sidebar.selectbox("Select ID", raw_df[id_col_long].unique())
            df_filtered = raw_df[raw_df[id_col_long] == selected_id_long].copy()
        else:
            df_filtered = raw_df.copy()
            
        # Parse Dates
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
        df_filtered = df_filtered.sort_values(date_col)
        
        series_data = df_filtered[target_col].values.astype(float)
        final_date_index = df_filtered[date_col]

# ==========================================
# 4. FORECASTING ENGINE
# ==========================================
if series_data is not None:
    # Visualization Area
    st.subheader(f"Historical Data ({len(series_data)} days)")
    
    # Forecast Settings
    prediction_length = st.slider("Forecast Horizon (Days)", 7, 60, 28)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Running Chronos-Bolt..."):
            # Prepare Tensor
            context_tensor = torch.tensor(series_data, dtype=torch.float32)
            
            # Predict
            batch_quantiles, _ = pipeline.predict_quantiles(
                inputs=[context_tensor],
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            
            # Extract
            forecast = batch_quantiles[0].numpy()
            low, median, high = forecast[:, 0], forecast[:, 1], forecast[:, 2]
            
            # Create Future Dates
            last_date = final_date_index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_length)
            
            # Plot
            fig = go.Figure()
            
            # Historical (Show last 100 points only to avoid lag)
            show_history = 100
            fig.add_trace(go.Scatter(
                x=final_date_index[-show_history:], 
                y=series_data[-show_history:],
                name="History", line=dict(color="gray")
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates, y=median,
                name="Forecast", line=dict(color="blue", width=2)
            ))
            
            # Uncertainty
            fig.add_trace(go.Scatter(
                x=future_dates, y=high,
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=low,
                name="90% Confidence",
                fill='tonexty', line=dict(width=0),
                fillcolor='rgba(0, 100, 255, 0.2)'
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("Predicted Demand", f"{int(median.sum())}")
            c2.metric("Safety Stock (P90)", f"{int(high.sum())}")

elif uploaded_file is None:
    st.info("👈 Upload your M5 `sales_train_evaluation.csv` in the sidebar and select 'Wide Format'!")