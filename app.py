import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Price Elasticity Simulator", layout="wide")
st.title("📈 Dynamic Pricing & Elasticity Simulator")
st.markdown("Optimize revenue and profit using a **Log-Log Elasticity Model**.")

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def get_sample_data():
    return pd.DataFrame({
        'Price': [10, 12, 15, 18, 20, 22, 25, 30, 35, 40],
        'Quantity': [500, 450, 380, 310, 290, 240, 200, 150, 100, 80]
    })

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    return output.getvalue()

def validate_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Dataset is empty.")
    if 'Price' not in data.columns or 'Quantity' not in data.columns:
        raise ValueError("Dataset must contain 'Price' and 'Quantity' columns.")

def calculate_elasticity(data):
    validate_data(data)

    clean_data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()
    if len(clean_data) < 2:
        raise ValueError("Need at least 2 valid positive observations.")

    X = np.log(clean_data['Price'])
    y = np.log(clean_data['Quantity'])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    beta = model.params.iloc[1]
    r_sq = model.rsquared
    p_value = model.pvalues.iloc[1]
    conf_int = model.conf_int().iloc[1]

    return beta, r_sq, model, p_value, conf_int

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("1️⃣ Data Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.divider()
st.sidebar.write("Need a template?")
st.sidebar.download_button(
    label="📥 Download Sample Template",
    data=to_excel(get_sample_data()),
    file_name="pricing_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -----------------------------
# Load Data
# -----------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"Data loaded successfully! Rows: {len(df)}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = get_sample_data()
else:
    st.info("Using sample dataset. Upload your own data in sidebar.")
    df = get_sample_data()

# -----------------------------
# Main Analysis
# -----------------------------
try:
    beta, r_sq, model, p_value, conf_int = calculate_elasticity(df)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Elasticity (β)", f"{beta:.4f}")
    col2.metric("R-Squared", f"{r_sq:.2%}")
    col3.metric("p-value", f"{p_value:.4f}")
    col4.metric("Market Type", "Elastic" if abs(beta) > 1 else "Inelastic")

    st.caption(f"95% Confidence Interval: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]")

    # -----------------------------
    # Strategic Recommendation
    # -----------------------------
    st.subheader("📋 Strategic Recommendation")

    if beta > -1:
        st.success(
            f"""
**Inelastic Market**

• 1% price increase → {abs(beta):.2f}% drop in volume  
• Customers are not very price sensitive  
• Consider controlled price increases to boost margins
"""
        )
    elif -2.5 <= beta <= -1:
        st.warning(
            f"""
**Elastic Market**

• 1% price increase → {abs(beta):.2f}% drop in volume  
• Customers are price sensitive  
• Promotional pricing may increase total revenue
"""
        )
    else:
        st.error(
            f"""
**Hyper-Elastic Market**

• 1% price increase → {abs(beta):.2f}% drop in volume  
• Commodity-like behavior  
• Focus on cost leadership & efficiency
"""
        )

    # -----------------------------
    # Simulation
    # -----------------------------
    st.divider()
    st.header("🕹️ What-If Simulation")

    col_sim1, col_sim2 = st.columns([1, 2])

    avg_q = df['Quantity'].mean()

    with col_sim1:
        current_price = st.number_input("Current Avg Price ($)", value=float(df['Price'].mean()), min_value=0.01)
        unit_cost = st.number_input("Unit Cost ($)", value=float(df['Price'].min() * 0.5), min_value=0.0)

        optimal_p = None
        if beta < -1:
            optimal_p = (unit_cost * beta) / (1 + beta)
            if optimal_p > 0 and np.isfinite(optimal_p):
                st.success(f"💡 Profit-Maximizing Price: ${optimal_p:.2f}")

        price_change = st.slider("Adjust Price (%)", -50, 100, 0, step=5)
        new_price = current_price * (1 + price_change / 100)

        predicted_q = avg_q * (new_price / current_price) ** beta
        new_revenue = new_price * predicted_q
        new_profit = (new_price - unit_cost) * predicted_q

        old_revenue = current_price * avg_q
        old_profit = (current_price - unit_cost) * avg_q

        st.write("### Predictions")
        st.write(f"Revenue: ${new_revenue:,.2f}")
        st.write(f"Profit: ${new_profit:,.2f}")
        st.write(f"Revenue Change: {((new_revenue-old_revenue)/old_revenue)*100:+.1f}%")
        if old_profit != 0:
            st.write(f"Profit Change: {((new_profit-old_profit)/old_profit)*100:+.1f}%")

    with col_sim2:
        prices = np.linspace(max(df['Price'].min()*0.5, 0.01), df['Price'].max()*1.5, 60)
        quantities = avg_q * (prices / current_price) ** beta
        revenues = prices * quantities
        profits = (prices - unit_cost) * quantities

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue"))
        fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit"))
        fig.add_vline(x=new_price, line_dash="dash", line_color="red", annotation_text="Your Price")

        if optimal_p and optimal_p > 0:
            fig.add_vline(x=optimal_p, line_dash="dot", line_color="green", annotation_text="Optimal Price")

        fig.update_layout(title="Revenue vs Profit Optimization", xaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Demand Curve
    # -----------------------------
    st.divider()
    st.subheader("📊 Demand Curve")

    fig_scatter = px.scatter(
        df,
        x="Price",
        y="Quantity",
        trendline="ols",
        title="Historical Price vs Quantity"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------
    # Model Summary
    # -----------------------------
    with st.expander("📋 Full Regression Summary"):
        st.write(model.summary())

    with st.expander("📈 Raw Data"):
        st.dataframe(df, use_container_width=True)

except Exception as e:
    st.error(f"Error in analysis: {e}")
    logger.error(e)
