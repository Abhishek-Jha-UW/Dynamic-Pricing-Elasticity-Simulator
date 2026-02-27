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
st.set_page_config(page_title="Price Elasticity Simulator", layout="wide", page_icon="📈")

# --- Custom Styling for a Professional Look ---
st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef;}
    div.stButton > button:first-child {background-color: #0083B8; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Dynamic Pricing & Elasticity Simulator")
st.markdown("Optimize your revenue and profit using **Log-Log Regression** models to understand demand sensitivity.")
st.divider()

# --- Helper Functions ---
@st.cache_data
def get_sample_data():
    """Generates realistic sample data following a power law with noise."""
    np.random.seed(42)
    # Simulate 20 price points
    prices = np.linspace(10, 50, 20)
    # True elasticity: -1.5 (Elastic), with random noise added
    noise = np.random.uniform(0.9, 1.1, 20)
    quantity = 1000 * (prices ** -1.5) * noise
    
    return pd.DataFrame({
        'Price': np.round(prices, 2),
        'Quantity': np.round(quantity)
    })

def to_excel(df):
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='SalesData')
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

    # Filter for positive values for log transformation
    clean_data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()

    if len(clean_data) < 3:
        raise ValueError("Not enough valid data points. Need at least 3 for reliable regression.")

    # Log-Log Regression for Constant Elasticity
    X = np.log(clean_data['Price'])
    y = np.log(clean_data['Quantity'])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    beta = model.params.iloc[1] # The elasticity coefficient
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
    st.info("👋 **Using sample data.** Upload your own in the sidebar to begin analysis.")
    df = get_sample_data()

# --- Main Analysis ---
if df is not None:
    try:
        beta, r_sq, model = calculate_elasticity(df)

        # --- Metrics ---
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Price Elasticity (β)", f"{beta:.4f}",
                      help="% change in quantity for 1% change in price. If β=-1.5, a 1% price increase drops volume by 1.5%.")

        with col2:
            st.metric("Model Fit (R-Squared)", f"{r_sq:.2%}",
                      help="Proportion of variance explained by the price-demand relationship.")

        with col3:
            status = "Elastic" if abs(beta) > 1 else "Inelastic"
            st.metric("Market Type", status)
            
        with col4:
            st.metric("Avg Price", f"${df['Price'].mean():.2f}")

        # --- Strategic Recommendation ---
        st.divider()
        st.subheader("💡 Strategic Recommendation")

        if beta > -1.0:
            st.success("💰 **Inelastic Market** (Customers are not very price sensitive)")
            st.markdown(f"""
            * **Price Impact:** A 1% increase in price leads to only a **{abs(beta):.2f}%** drop in volume.
            * **Action:** Consider controlled price increases to improve margins without significant volume loss.
            """)

        elif -2.5 <= beta <= -1.0:
            st.warning("📊 **Elastic Market** (Customers are price sensitive)")
            st.markdown(f"""
            * **Price Impact:** A 1% increase in price leads to a **{abs(beta):.2f}%** drop in volume.
            * **Action:** Use promotional pricing carefully. A small price reduction may increase total revenue through higher volume.
            """)

        else:
            st.error("📉 **Hyper-Elastic Market** (Commodity-like dynamics)")
            st.markdown(f"""
            * **Price Impact:** A 1% increase in price leads to a significant **{abs(beta):.2f}%** drop in volume.
            * **Action:** Focus on operational efficiency and lowering unit costs to remain competitive.
            """)

        # --- Simulation ---
        st.divider()
        st.header("🕹️ What-If Simulation")
        st.markdown("Adjust parameters to forecast revenue and profit.")

        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            current_price = st.number_input(
                "Current Avg Price ($)",
                value=float(df['Price'].mean()),
                min_value=0.01
            )

            unit_cost = st.number_input(
                "Unit Cost ($)",
                value=float(df['Price'].min() * 0.7),
                min_value=0.0
            )

            # Profit Maximizing Price formula for Constant Elasticity
            if beta < -1:
                optimal_p = (unit_cost * beta) / (1 + beta)
                if optimal_p > 0:
                    st.success(f"🎯 Profit-Maximizing Price: **${optimal_p:.2f}**")

            price_change = st.slider("Adjust Price (%)", -50, 100, 0, step=5)

            new_price = current_price * (1 + price_change / 100)
            # Q2 = Q1 * (P2/P1)^beta
            predicted_q = df['Quantity'].mean() * (new_price / current_price) ** beta
            
            new_revenue = new_price * predicted_q
            new_profit = (new_price - unit_cost) * predicted_q

            old_revenue = current_price * df['Quantity'].mean()
            old_profit = (current_price - unit_cost) * df['Quantity'].mean()

            st.write("### 📊 Projections")
            st.metric("Predicted Revenue", f"${new_revenue:,.2f}", f"{((new_revenue-old_revenue)/old_revenue)*100:+.1f}% vs Current")
            st.metric("Predicted Profit", f"${new_profit:,.2f}", f"{((new_profit-old_profit)/old_profit)*100:+.1f}% vs Current")

        with col_sim2:
            # Generate range for curve
            prices = np.linspace(max(df['Price'].min()*0.5, 0.01),
                                 df['Price'].max()*1.5, 100)

            quantities = df['Quantity'].mean() * (prices / current_price) ** beta
            revenues = prices * quantities
            profits = (prices - unit_cost) * quantities

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue", line=dict(color='#0083B8')))
            fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit", line=dict(color='#28a745', dash='dash')))
            fig.add_vline(x=new_price, line_dash="dash",
                          line_color="red", annotation_text="Simulated Price")

            fig.update_layout(
                title="Revenue vs Profit Optimization Curve",
                xaxis_title="Price ($)",
                yaxis_title="USD ($)",
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- Model Diagnostics & Data ---
        st.divider()
        with st.expander("📊 View Model Diagnostics & Data"):

            st.subheader("Historical Demand Plot")
            # Plot raw data with the fitted regression line
            fig_scatter = px.scatter(
                df,
                x="Price",
                y="Quantity",
                title="Historical Sales Data",
                labels={"Price": "Price ($)", "Quantity": "Units Sold"}
            )
            # Add the modeled curve
            fig_scatter.add_trace(go.Scatter(x=prices, y=quantities, name="Log-Log Fit", line=dict(color='red')))
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Detailed Regression Summary")
            st.write(model.summary())
            
            st.subheader("Raw Data")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error in analysis: {e}")
        logger.error(e)
