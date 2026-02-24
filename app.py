import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Price Elasticity Simulator", layout="wide")
st.title("📈 Dynamic Pricing & Elasticity Simulator")
st.markdown("Optimize your revenue and profit using Log-Log Regression models.")

# --- Helper Functions ---
@st.cache_data
def get_sample_data():
    return pd.DataFrame({
        'Price': [10, 12, 15, 18, 20, 22, 25, 30, 35, 40],
        'Quantity': [500, 450, 380, 310, 290, 240, 200, 150, 100, 80]
    })

def to_excel(df):
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error converting to Excel: {e}")
        st.error("Failed to generate Excel file")
        return None

def validate_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Data is empty")
    if 'Price' not in data.columns or 'Quantity' not in data.columns:
        raise ValueError("Data must contain 'Price' and 'Quantity' columns")
    return True

def calculate_elasticity(data):
    validate_data(data)

    clean_data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()

    if len(clean_data) < 2:
        raise ValueError("Not enough valid data points. Need at least 2.")

    X = np.log(clean_data['Price'])
    y = np.log(clean_data['Quantity'])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[1]
    r_sq = model.rsquared

    return beta, r_sq, model

# --- Sidebar ---
st.sidebar.header("1. Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV or Excel)", type=["csv", "xlsx"])

st.sidebar.divider()
st.sidebar.write("💡 **Need a template?**")

excel_data = to_excel(get_sample_data())
if excel_data:
    st.sidebar.download_button(
        label="📥 Download Excel Template",
        data=excel_data,
        file_name='pricing_template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# --- Load Data ---
df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Data loaded successfully! Rows: {len(df)}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        df = get_sample_data()
else:
    st.info("👋 Using sample data. Upload your own in the sidebar to begin.")
    df = get_sample_data()

# --- Main Analysis ---
if df is not None:
    try:
        beta, r_sq, model = calculate_elasticity(df)

        # --- Metrics ---
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Price Elasticity (β)", f"{beta:.4f}",
                      help="% change in quantity for 1% change in price")

        with col2:
            st.metric("R-Squared (Model Fit)", f"{r_sq:.2%}",
                      help="Proportion of variance explained")

        with col3:
            status = "📈 Elastic" if abs(beta) > 1 else "📉 Inelastic"
            st.metric("Market Type", status)

        # --- Strategic Recommendation ---
        st.subheader("Strategic Recommendation")

        if beta > -1.0:
            st.success("💰 Inelastic Market")
            st.markdown(f"""
**1️⃣ Price Impact**  
A 1% increase in price leads to only a **{abs(beta):.2f}%** drop in volume.

**2️⃣ Market Behavior**  
Demand is **Inelastic**. Customers are relatively insensitive to price changes.

**3️⃣ Strategic Action**  
Consider controlled price increases to improve margins without significant volume loss.
""")

        elif -2.5 <= beta <= -1.0:
            st.warning("📊 Elastic Market")
            st.markdown(f"""
**1️⃣ Price Impact**  
A 1% increase in price leads to a **{abs(beta):.2f}%** drop in volume.

**2️⃣ Market Behavior**  
Demand is **Elastic**. Customers are price sensitive.

**3️⃣ Strategic Action**  
Use promotional pricing carefully. A price reduction may increase total revenue through higher volume.
""")

        else:
            st.error("📉 Hyper-Elastic Market")
            st.markdown(f"""
**1️⃣ Price Impact**  
A 1% increase in price leads to a significant **{abs(beta):.2f}%** drop in volume.

**2️⃣ Market Behavior**  
Demand is **Highly Elastic**, indicating commodity-like dynamics.

**3️⃣ Strategic Action**  
Focus on operational efficiency and lowering unit costs to remain competitive.
""")

        # --- Simulation ---
        st.divider()
        st.header("🕹️ What-If Simulation")

        col_sim1, col_sim2 = st.columns([1, 2])
        avg_q = df['Quantity'].mean()

        with col_sim1:
            current_price = st.number_input(
                "Current Avg Price ($)",
                value=float(df['Price'].mean()),
                min_value=0.01
            )

            unit_cost = st.number_input(
                "Unit Cost ($)",
                value=float(df['Price'].min() * 0.5),
                min_value=0.0
            )

            if beta < -1:
                optimal_p = (unit_cost * beta) / (1 + beta)
                if optimal_p > 0:
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

            st.write("### Changes")
            st.write(f"Revenue Change: {((new_revenue-old_revenue)/old_revenue)*100:+.1f}%")

            if old_profit != 0:
                st.write(f"Profit Change: {((new_profit-old_profit)/old_profit)*100:+.1f}%")

        with col_sim2:
            prices = np.linspace(max(df['Price'].min()*0.5, 0.01),
                                 df['Price'].max()*1.5, 50)

            quantities = avg_q * (prices / current_price) ** beta
            revenues = prices * quantities
            profits = (prices - unit_cost) * quantities

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue"))
            fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit"))
            fig.add_vline(x=new_price, line_dash="dash",
                          line_color="red", annotation_text="Your Price")

            fig.update_layout(
                title="Revenue vs Profit Optimization",
                xaxis_title="Price ($)",
                yaxis_title="Total Value ($)",
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- Combined Model Diagnostics Dropdown ---
        st.divider()
        with st.expander("📊 Model Diagnostics & Historical Demand Analysis"):

            st.subheader("Demand Curve & Historical Data")

            fig_scatter = px.scatter(
                df,
                x="Price",
                y="Quantity",
                trendline="ols",
                title="Historical Demand Relationship (Log-Log Fit)",
                labels={"Price": "Price ($)", "Quantity": "Quantity (units)"}
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Detailed Regression Summary")
            st.write(model.summary())

        # --- Raw Data ---
        with st.expander("📈 View Raw Data"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error in analysis: {e}")
        logger.error(e)
