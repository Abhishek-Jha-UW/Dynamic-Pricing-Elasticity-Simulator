import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Price Elasticity Simulator", layout="wide")
st.title("Dynamic Pricing & Elasticity Simulator")
st.markdown("Optimize your revenue and profit using Log-Log Regression models.")

# --- Sidebar: Data Upload ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Sales CSV (Columns: Price, Quantity)", type="csv")

# Sample data generation if no file is uploaded
if uploaded_file is None:
    st.info("Upload a CSV to start. Using sample data for demonstration.")
    df = pd.DataFrame({
        'Price': np.linspace(10, 50, 20),
        'Quantity': [100, 92, 85, 70, 65, 58, 50, 42, 38, 30, 28, 22, 20, 15, 12, 10, 8, 5, 4, 2]
    })
else:
    df = pd.read_csv(uploaded_file)

# --- Core Logic: Log-Log Regression ---
def calculate_elasticity(data):
    # Ensure no zeros for log transformation
    data = data[(data['Price'] > 0) & (data['Quantity'] > 0)]
    
    X = np.log(data['Price'])
    y = np.log(data['Quantity'])
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    beta = model.params[1]  # The Elasticity Coefficient
    r_sq = model.rsquared
    return beta, r_sq, model

beta, r_sq, model = calculate_elasticity(df)

# --- UI Layout: Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Price Elasticity (β)", f"{beta:.2f}")
col2.metric("R-Squared (Model Fit)", f"{r_sq:.2%}")
status = "Elastic" if abs(beta) > 1 else "Inelastic"
col3.metric("Market Type", status)



# --- Simulator Section ---
st.divider()
st.header("🕹️ 'What-If' Simulation")

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.subheader("Control Parameters")
    current_price = st.number_input("Current Avg Price ($)", value=float(df['Price'].mean()))
    unit_cost = st.number_input("Unit Cost ($)", value=float(df['Price'].min() * 0.5))
    price_change = st.slider("Adjust Price (%)", -50, 100, 0)
    
    new_price = current_price * (1 + price_change / 100)
    # Predict new quantity: Q2 = Q1 * (P2/P1)^beta
    avg_q = df['Quantity'].mean()
    predicted_q = avg_q * (new_price / current_price)**beta
    
    new_revenue = new_price * predicted_q
    new_profit = (new_price - unit_cost) * predicted_q
    
    st.write(f"**Predicted Quantity:** {predicted_q:.1f} units")
    st.write(f"**Predicted Revenue:** ${new_revenue:,.2f}")
    st.write(f"**Predicted Profit:** ${new_profit:,.2f}")

with col_sim2:
    # Optimization Chart
    prices = np.linspace(df['Price'].min(), df['Price'].max() * 1.5, 50)
    quantities = avg_q * (prices / current_price)**beta
    revenues = prices * quantities
    profits = (prices - unit_cost) * quantities

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue", line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit", line=dict(color='green', width=3)))
    fig.add_vline(x=new_price, line_dash="dash", line_color="red", annotation_text="Your Price")
    
    fig.update_layout(title="Revenue vs Profit Optimization", xaxis_title="Price ($)", yaxis_title="Total Value ($)")
    st.plotly_chart(fig, use_container_width=True)



st.divider()
st.subheader("Raw Data & Demand Curve")
fig_scatter = px.scatter(df, x="Price", y="Quantity", trendline="ols", title="Historical Demand Relationship")
st.plotly_chart(fig_scatter, use_container_width=True)
