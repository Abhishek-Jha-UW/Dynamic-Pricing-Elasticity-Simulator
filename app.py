import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import logging

# ======================================================
# CONFIGURATION
# ======================================================
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

# ======================================================
# HELPER FUNCTIONS
# ======================================================

@st.cache_data
def get_sample_data():
    """Provides demonstration dataset."""
    return pd.DataFrame({
        'Price': [10, 12, 15, 18, 20, 22, 25, 30, 35, 40],
        'Quantity': [500, 450, 380, 310, 290, 240, 200, 150, 100, 80]
    })

def validate_and_clean(data):
    if data is None or len(data) < 2:
        raise ValueError("Dataset must contain at least two rows.")
    if not {'Price', 'Quantity'}.issubset(data.columns):
        raise ValueError("Data must contain 'Price' and 'Quantity' columns.")
    
    clean = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()
    
    if len(clean) < 2:
        raise ValueError("Insufficient positive values for log-log model.")
    
    return clean

def fit_model(data):
    clean = validate_and_clean(data)

    X = np.log(clean['Price'])
    y = np.log(clean['Quantity'])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    return model, alpha, beta

def predict_quantity(alpha, beta, price):
    return np.exp(alpha) * price ** beta

# ======================================================
# SIDEBAR – DATA INPUT
# ======================================================

st.sidebar.header("Data Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.sidebar.success("Data loaded successfully.")
    except Exception as e:
        logger.error(e)
        st.sidebar.warning("Error loading file. Using sample data.")
        df = get_sample_data()
else:
    df = get_sample_data()
    st.sidebar.info("Using sample dataset.")

# ======================================================
# MAIN EXECUTION
# ======================================================

try:
    model, alpha, beta = fit_model(df)

    r_sq = model.rsquared
    p_val = model.pvalues.iloc[1]
    conf_int = model.conf_int().iloc[1]

    # ==================================================
    # ELASTICITY RESULTS
    # ==================================================

    st.subheader("Elasticity Estimation Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Elasticity (β)", f"{beta:.3f}")
    c2.metric("Model R²", f"{r_sq:.2%}")
    c3.metric("p-value", f"{p_val:.4f}")
    c4.metric("95% CI", f"[{conf_int[0]:.2f}, {conf_int[1]:.2f}]")

    # Market classification
    if beta < -1:
        market_type = "Elastic"
    elif -1 <= beta < 0:
        market_type = "Inelastic"
    else:
        market_type = "Upward Sloping (Investigate Data)"

    st.markdown(
        f"""
### Interpretation
- A 1% increase in price leads to an estimated **{abs(beta):.2f}% change** in quantity demanded.
- Demand is classified as **{market_type}**.
- Elasticity is {'statistically significant' if p_val < 0.05 else 'not statistically significant'} at the 5% level.
"""
    )

    # ==================================================
    # SCENARIO SIMULATION
    # ==================================================

    st.divider()
    st.header("Pricing Scenario Simulation")

    col_left, col_right = st.columns([1, 2])

    with col_left:

        current_price = st.number_input(
            "Current Price ($)",
            value=float(df['Price'].mean()),
            min_value=0.01
        )

        unit_cost = st.number_input(
            "Unit Variable Cost ($)",
            value=float(df['Price'].min() * 0.5),
            min_value=0.0
        )

        price_change = st.slider(
            "Price Adjustment (%)",
            -50, 100, 0, step=5
        )

        new_price = current_price * (1 + price_change / 100)

        current_q = predict_quantity(alpha, beta, current_price)
        new_q = predict_quantity(alpha, beta, new_price)

        current_revenue = current_price * current_q
        new_revenue = new_price * new_q

        current_profit = (current_price - unit_cost) * current_q
        new_profit = (new_price - unit_cost) * new_q

        st.subheader("Projected Impact")

        m1, m2 = st.columns(2)

        revenue_delta = ((new_revenue/current_revenue - 1) * 100) if current_revenue != 0 else 0
        profit_delta = ((new_profit/current_profit - 1) * 100) if current_profit != 0 else 0

        m1.metric("Revenue", f"${new_revenue:,.2f}", delta=f"{revenue_delta:.1f}%")
        m2.metric("Profit", f"${new_profit:,.2f}", delta=f"{profit_delta:.1f}%")

    with col_right:

        prices = np.linspace(df['Price'].min() * 0.5,
                             df['Price'].max() * 1.5, 200)

        quantities = predict_quantity(alpha, beta, prices)
        revenues = prices * quantities
        profits = (prices - unit_cost) * quantities

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue"))
        fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit"))

        fig.add_vline(x=current_price, line_dash="solid", annotation_text="Baseline")
        fig.add_vline(x=new_price, line_dash="dash", annotation_text="Scenario")

        # Profit-maximizing price (only if β < -1)
        if beta < -1:
            optimal_price = (unit_cost * beta) / (1 + beta)
            if optimal_price > 0:
                fig.add_vline(x=optimal_price, line_dash="dot",
                              annotation_text="Profit-Max")

        fig.update_layout(
            title="Revenue and Profit vs Price",
            xaxis_title="Price ($)",
            yaxis_title="Value ($)",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # REVENUE MAX INSIGHT
    # ==================================================

    st.divider()
    st.subheader("Revenue Maximization Insight")

    if abs(beta + 1) < 0.1:
        st.success("Demand is near unit elastic. Revenue is approximately maximized.")
    elif beta < -1:
        st.info("Demand is elastic. Lowering price may increase total revenue.")
    elif -1 <= beta < 0:
        st.info("Demand is inelastic. Increasing price may increase total revenue.")
    else:
        st.warning("Upward sloping demand detected. Verify data validity.")

    # ==================================================
    # DIAGNOSTICS
    # ==================================================

    with st.expander("Model Diagnostics"):

        st.write("### Regression Summary")
        st.write(model.summary())

        residuals = model.resid

        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=df['Price'],
            y=residuals,
            mode="markers",
            name="Residuals"
        ))

        fig_resid.update_layout(
            title="Residuals vs Price",
            xaxis_title="Price",
            yaxis_title="Residual"
        )

        st.plotly_chart(fig_resid, use_container_width=True)

    # ==================================================
    # ASSUMPTIONS & LIMITATIONS
    # ==================================================

    with st.expander("Model Assumptions & Limitations"):
        st.markdown("""
- Constant elasticity demand assumption
- No competitor reaction modeled
- No seasonality or time effects included
- Assumes price is exogenous
- Log-log linear relationship
""")

    # ==================================================
    # RAW DATA
    # ==================================================

    with st.expander("Raw Data"):
        st.dataframe(df, use_container_width=True)

except Exception as e:
    logger.error(e)
    st.error(f"Application error: {e}")
