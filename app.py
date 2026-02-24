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
    """Generate sample data for demonstration."""
    return pd.DataFrame({
        'Price': [10, 12, 15, 18, 20, 22, 25, 30, 35, 40],
        'Quantity': [500, 450, 380, 310, 290, 240, 200, 150, 100, 80]
    })

def to_excel(df):
    """Convert DataFrame to Excel bytes."""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error converting to Excel: {e}")
        return None

def validate_data(data):
    """Validate input data."""
    if data is None or len(data) == 0:
        raise ValueError("Data is empty")
    if 'Price' not in data.columns or 'Quantity' not in data.columns:
        raise ValueError("Data must contain 'Price' and 'Quantity' columns")
    return True

def calculate_elasticity(data):
    """Calculate price elasticity using Log-Log Regression."""
    validate_data(data)
    clean_data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()
    
    if len(clean_data) < 2:
        raise ValueError(f"Need at least 2 valid data points, got {len(clean_data)}")
    
    X = np.log(clean_data['Price'])
    y = np.log(clean_data['Quantity'])
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    r_sq = model.rsquared
    return beta, r_sq, model

# --- Sidebar: Data Management ---
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

# --- Data Selection Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Data loaded! Rows: {len(df)}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        df = get_sample_data()
else:
    st.info("👋 Use the sidebar to upload data or explore with our sample set.")
    df = get_sample_data()

# --- Main Analysis ---
if df is not None:
    try:
        beta, r_sq, model = calculate_elasticity(df)
        
        # 1. Key Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Price Elasticity (β)", f"{beta:.4f}")
        m2.metric("R-Squared (Model Fit)", f"{r_sq:.2%}")
        status = "📈 Elastic" if abs(beta) > 1 else "📉 Inelastic"
        m3.metric("Market Type", status)

        # 2. Strategic Recommendations (Your Revised 2-Point Section)
        st.subheader("📋 Strategic Recommendations")
        
        if beta > -1.0:
            market_type, st_box = "Inelastic", st.success
            advice = "Customers aren't very price-sensitive. You have 'Pricing Power.' Consider small price increases."
        elif -2.5 <= beta <= -1.0:
            market_type, st_box = "Elastic", st.warning
            advice = "Customers are sensitive. Focus on promotional pricing. A price cut could lead to a 'Sales Surge' that outweighs the lower margin."
        else:
            market_type, st_box = "Hyper-Elastic", st.error
            advice = "You are in a commodity market. Focus on operational efficiency and lowering Unit Cost."

        st_box(f"""
        1. **Price Impact:** A 1% price increase is leading to a **{abs(beta):.2f}%** change in sales volume.
        2. **Market Insight:** Demand is **{market_type}**. {advice}
        """)

        # 3. What-If Simulation
        st.divider()
        st.header("🕹️ What-If Simulation")
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.subheader("Control Parameters")
            current_price = st.number_input("Current Avg Price ($)", value=float(df['Price'].mean()), step=1.0)
            unit_cost = st.number_input("Unit Cost ($)", value=float(df['Price'].min() * 0.5), step=0.5)
            
            # Profit-Maximizing Price Logic
            if beta < -1:
                optimal_p = (unit_cost * beta) / (1 + beta)
                if optimal_p > 0:
                    st.success(f"💡 **Profit-Maximizing Price: ${optimal_p:.2f}**")
            
            price_change = st.slider("Adjust Price (%)", -50, 100, 0, step=5)
            new_price = current_price * (1 + price_change / 100)
            avg_q = df['Quantity'].mean()
            predicted_q = avg_q * (new_price / current_price) ** beta if current_price > 0 else 0
            
            st.write("### Predictions")
            st.write(f"**New Price:** ${new_price:.2f}")
            st.write(f"**Predicted Quantity:** {predicted_q:.1f} units")
            st.write(f"**Predicted Profit:** ${(new_price - unit_cost) * predicted_q:,.2f}")

        with col_sim2:
            prices = np.linspace(max(df['Price'].min() * 0.5, 0.1), df['Price'].max() * 1.5, 50)
            quantities = avg_q * (prices / current_price) ** beta
            profits = (prices - unit_cost) * quantities
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=prices * quantities, name="Revenue", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit", line=dict(color='green')))
            fig.add_vline(x=new_price, line_dash="dash", line_color="red", annotation_text="Your Price")
            fig.update_layout(title="Revenue vs Profit Optimization", xaxis_title="Price ($)", height=450)
            st.plotly_chart(fig, use_container_width=True)

        # 4. Detailed Charts
        with st.expander("📊 View Demand Curve & Raw Data"):
            fig_scatter = px.scatter(df, x="Price", y="Quantity", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error in analysis: {e}")
