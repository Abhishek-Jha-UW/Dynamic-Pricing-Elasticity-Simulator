import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import logging

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Price Elasticity & Profit Optimization",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Price Elasticity & Profit Optimization Dashboard")
st.markdown(
"""
This dashboard estimates **price elasticity of demand** using a log-log regression model  
and simulates revenue and profit outcomes under alternative pricing strategies.
"""
)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
@st.cache_data
def get_sample_data():
    return pd.DataFrame({
        'Price': [10, 12, 15, 18, 20, 22, 25, 30, 35, 40],
        'Quantity': [500, 450, 380, 310, 290, 240, 200, 150, 100, 80]
    })

def validate_data(data):
    if data is None or len(data) < 2:
        raise ValueError("Dataset must contain at least two rows.")
    if not {'Price', 'Quantity'}.issubset(data.columns):
        raise ValueError("Data must contain 'Price' and 'Quantity' columns.")
    return data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()

def fit_model(data):
    clean_data = validate_data(data)

    X = np.log(clean_data['Price'])
    y = np.log(clean_data['Quantity'])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return model, alpha, beta

def predict_quantity(alpha, beta, price):
    return np.exp(alpha) * price ** beta

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.sidebar.success("Data loaded successfully.")
    except:
        st.sidebar.warning("Error loading file. Using sample data.")
        df = get_sample_data()
else:
    df = get_sample_data()
    st.sidebar.info("Using sample dataset.")

# -------------------------------------------------------
# Main Analysis
# -------------------------------------------------------
try:
    model, alpha, beta = fit_model(df)
    r_sq = model.rsquared
    p_val = model.pvalues.iloc[1]
    conf_int = model.conf_int().iloc[1]

    # ---------------------------------------------------
    # Model Metrics
    # ---------------------------------------------------
    st.subheader("Elasticity Estimation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Elasticity (β)", f"{beta:.3f}")
    col2.metric("R²", f"{r_sq:.2%}")
    col3.metric("p-value", f"{p_val:.4f}")
    col4.metric("95% CI", f"[{conf_int[0]:.2f}, {conf_int[1]:.2f}]")

    # Market Type
    if beta < -1:
        market_type = "Elastic"
    elif -1 <= beta < 0:
        market_type = "Inelastic"
    else:
        market_type = "Upward Sloping"

    st.markdown(
        f"""
**Interpretation**

- A 1% increase in price leads to an estimated **{abs(beta):.2f}% change** in demand.
- Demand is classified as **{market_type}**.
- The elasticity estimate is statistically {'significant' if p_val < 0.05 else 'not statistically significant'} at 5% level.
"""
    )

    # ---------------------------------------------------
    # Scenario Simulation
    # ---------------------------------------------------
    st.divider()
    st.header("Pricing Scenario Simulation")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        current_price = st.number_input("Current Price", value=float(df['Price'].mean()), min_value=0.01)
        unit_cost = st.number_input("Unit Cost", value=float(df['Price'].min()*0.5), min_value=0.0)
        price_change = st.slider("Price Adjustment (%)", -50, 100, 0, step=5)

        new_price = current_price * (1 + price_change/100)

        current_q = predict_quantity(alpha, beta, current_price)
        new_q = predict_quantity(alpha, beta, new_price)

        current_revenue = current_price * current_q
        new_revenue = new_price * new_q

        current_profit = (current_price - unit_cost) * current_q
        new_profit = (new_price - unit_cost) * new_q

        colA, colB = st.columns(2)
        colA.metric("Revenue", f"${new_revenue:,.2f}",
                    delta=f"{(new_revenue/current_revenue - 1)*100:.1f}%")
        colB.metric("Profit", f"${new_profit:,.2f}",
                    delta=f"{(new_profit/current_profit - 1)*100:.1f}%")

    with col_right:
        prices = np.linspace(df['Price'].min()*0.5,
                             df['Price'].max()*1.5, 200)

        quantities = predict_quantity(alpha, beta, prices)
        revenues = prices * quantities
        profits = (prices - unit_cost) * quantities

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue"))
        fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit"))
        fig.add_vline(x=new_price, line_dash="dash", annotation_text="Selected Price")

        if beta < -1:
            optimal_price = (unit_cost * beta) / (1 + beta)
            if optimal_price > 0:
                fig.add_vline(x=optimal_price, line_dash="dot",
                              annotation_text="Profit-Max Price")

        fig.update_layout(
            xaxis_title="Price",
            yaxis_title="Value",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------
    with st.expander("Model Diagnostics"):
        st.write(model.summary())

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df['Price'],
            y=df['Quantity'],
            mode='markers',
            name="Observed"
        ))

        price_line = np.linspace(df['Price'].min(),
                                 df['Price'].max(), 200)

        q_line = predict_quantity(alpha, beta, price_line)

        fig_scatter.add_trace(go.Scatter(
            x=price_line,
            y=q_line,
            name="Fitted Curve"
        ))

        fig_scatter.update_layout(
            xaxis_title="Price",
            yaxis_title="Quantity"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(df, use_container_width=True)

except Exception as e:
    logger.error(e)
    st.error(f"Analysis failed: {e}")
