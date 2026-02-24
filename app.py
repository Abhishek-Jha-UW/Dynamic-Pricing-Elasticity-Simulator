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
        st.error("Failed to generate Excel file")
        return None

def validate_data(data):
    """Validate input data for elasticity calculation."""
    if data is None or len(data) == 0:
        raise ValueError("Data is empty")
    
    if 'Price' not in data.columns or 'Quantity' not in data.columns:
        raise ValueError("Data must contain 'Price' and 'Quantity' columns")
    
    # Check for negative or zero values
    if (data['Price'] <= 0).any() or (data['Quantity'] <= 0).any():
        logger.warning("Data contains zero or negative values - they will be filtered out")
    
    return True

def calculate_elasticity(data):
    """Calculate price elasticity using Log-Log Regression with error handling."""
    try:
        # Validate input
        validate_data(data)
        
        # Filter out zeros and negatives for log transformation
        clean_data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()
        
        # Check if we have enough data points
        if len(clean_data) < 2:
            raise ValueError(f"Not enough valid data points. Need at least 2, got {len(clean_data)}")
        
        # Log transformation
        X = np.log(clean_data['Price'])
        y = np.log(clean_data['Quantity'])
        X = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        beta = model.params[1]  # Elasticity coefficient
        r_sq = model.rsquared
        
        logger.info(f"Elasticity calculated: {beta:.4f}, R-squared: {r_sq:.4f}")
        return beta, r_sq, model
        
    except Exception as e:
        logger.error(f"Error calculating elasticity: {e}")
        raise

# --- Sidebar: Data Management ---
st.sidebar.header("1. Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV or Excel)", type=["csv", "xlsx"])

st.sidebar.divider()
st.sidebar.write("💡 **Need a template?**")
sample_template = get_sample_data()

excel_data = to_excel(sample_template)
if excel_data:
    st.sidebar.download_button(
        label="📥 Download Excel Template",
        data=excel_data,
        file_name='pricing_template.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# --- Data Selection Logic ---
df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        logger.info(f"File uploaded successfully: {uploaded_file.name}")
        st.success(f"✅ Data loaded successfully! Rows: {len(df)}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        logger.error(f"File upload error: {e}")
else:
    st.info("👋 **Welcome!** You can upload a file in the sidebar or use our sample data to explore.")
    if st.button("📊 Load Sample Data & Run Analysis"):
        df = get_sample_data()
        st.session_state['using_sample'] = True
    else:
        df = get_sample_data()  # Default fallback

# --- Main Analysis Section ---
if df is not None:
    try:
        # Calculate elasticity
        beta, r_sq, model = calculate_elasticity(df)
        
        # --- Display Key Metrics ---
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
            st.metric("Market Type", status,
                     help="Elastic: Q changes more than P; Inelastic: Q changes less than P")
        
        # --- Elasticity Interpretation ---
        st.info(f"""
        **Interpretation:** A 1% increase in price leads to a **{beta:.2f}%** change in quantity demanded.
        """)
        
        # --- Simulator Section ---
        st.divider()
        st.header("🕹️ What-If Simulation")
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.subheader("Control Parameters")
            current_price = st.number_input(
                "Current Avg Price ($)", 
                value=float(df['Price'].mean()),
                min_value=0.01,
                step=1.0
            )
            unit_cost = st.number_input(
                "Unit Cost ($)", 
                value=float(df['Price'].min() * 0.5),
                min_value=0.0,
                step=0.5
            )
            price_change = st.slider("Adjust Price (%)", -50, 100, 0, step=5)
            
            # Calculations
            new_price = current_price * (1 + price_change / 100)
            avg_q = df['Quantity'].mean()
            
            # Safe calculation with zero division check
            if current_price > 0:
                predicted_q = avg_q * (new_price / current_price) ** beta
            else:
                predicted_q = 0
            
            new_revenue = new_price * predicted_q
            new_profit = (new_price - unit_cost) * predicted_q
            
            # Display results with formatting
            st.write("### Predictions")
            st.write(f"**New Price:** ${new_price:.2f}")
            st.write(f"**Predicted Quantity:** {predicted_q:.1f} units")
            st.write(f"**Predicted Revenue:** ${new_revenue:,.2f}")
            st.write(f"**Predicted Profit:** ${new_profit:,.2f}")
            
            # Calculate % changes
            old_revenue = current_price * avg_q
            old_profit = (current_price - unit_cost) * avg_q
            revenue_change = ((new_revenue - old_revenue) / old_revenue * 100) if old_revenue > 0 else 0
            profit_change = ((new_profit - old_profit) / old_profit * 100) if old_profit > 0 else 0
            
            st.write("### Changes")
            st.write(f"Revenue Change: {revenue_change:+.1f}%")
            st.write(f"Profit Change: {profit_change:+.1f}%")
        
        with col_sim2:
            # Optimization Chart
            prices = np.linspace(max(df['Price'].min() * 0.5, 0.01), df['Price'].max() * 1.5, 50)
            quantities = avg_q * (prices / current_price) ** beta if current_price > 0 else avg_q * np.ones_like(prices)
            revenues = prices * quantities
            profits = (prices - unit_cost) * quantities
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices, y=revenues, 
                name="Revenue", 
                line=dict(color='blue', width=3),
                hovertemplate='Price: $%{x:.2f}<br>Revenue: $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=prices, y=profits, 
                name="Profit", 
                line=dict(color='green', width=3),
                hovertemplate='Price: $%{x:.2f}<br>Profit: $%{y:,.0f}<extra></extra>'
            ))
            fig.add_vline(x=new_price, line_dash="dash", line_color="red", 
                         annotation_text="Your Price", annotation_position="top right")
            
            fig.update_layout(
                title="Revenue vs Profit Optimization",
                xaxis_title="Price ($)",
                yaxis_title="Total Value ($)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Demand Curve ---
        st.divider()
        st.subheader("📊 Demand Curve & Historical Data")
        
        fig_scatter = px.scatter(
            df, x="Price", y="Quantity", 
            trendline="ols",
            title="Historical Demand Relationship (Log-Log Fit)",
            labels={"Price": "Price ($)", "Quantity": "Quantity (units)"},
            hover_data={"Price": ":.2f", "Quantity": ":.0f"}
        )
        fig_scatter.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # --- Model Summary ---
        with st.expander("📋 Detailed Model Summary"):
            st.write(model.summary())
        
        # --- Data Table ---
        with st.expander("📈 View Raw Data"):
            st.dataframe(df, use_container_width=True)
    
    except ValueError as e:
        st.error(f"❌ Data Validation Error: {e}")
        logger.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}")
