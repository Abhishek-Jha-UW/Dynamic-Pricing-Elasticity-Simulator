import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import logging

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Interfaces
# ---------------------------------------------------------
class DataInterface:
    """Handles sample data, uploads, validation, and export."""

    @staticmethod
    @st.cache_data
    def sample_data(n=50):
        # Generate synthetic log-log demand curve with noise
        prices = np.linspace(8, 40, n)
        elasticity = -1.4
        base_qty = 600

        quantities = base_qty * (prices / prices.mean()) ** elasticity
        noise = np.random.normal(0, quantities.std() * 0.05, n)
        quantities = np.maximum(quantities + noise, 1)

        return pd.DataFrame({"Price": prices, "Quantity": quantities.astype(int)})

    @staticmethod
    def load(uploaded_file):
        if uploaded_file is None:
            return DataInterface.sample_data()

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            st.error("Failed to load file. Using sample data instead.")
            return DataInterface.sample_data()

    @staticmethod
    def validate(df):
        if df is None or df.empty:
            raise ValueError("Data is empty.")
        if not {"Price", "Quantity"}.issubset(df.columns):
            raise ValueError("Data must contain Price and Quantity columns.")
        if (df["Price"] <= 0).any() or (df["Quantity"] <= 0).any():
            raise ValueError("Price and Quantity must be positive.")
        return True

    @staticmethod
    def to_excel(df):
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return None


class ModelInterface:
    """Handles elasticity modeling and predictions."""

    @staticmethod
    def fit(df):
        DataInterface.validate(df)
        df = df[(df["Price"] > 0) & (df["Quantity"] > 0)]

        X = np.log(df["Price"])
        y = np.log(df["Quantity"])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]
        r_sq = model.rsquared
        return beta, r_sq, model

    @staticmethod
    def predict_quantity(avg_q, price_ratio, beta):
        return avg_q * (price_ratio ** beta)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Price Elasticity Simulator", layout="wide")
st.title("📈 Dynamic Pricing & Elasticity Simulator")
st.markdown("Optimize revenue and profit using Log-Log Regression models.")

# Sidebar
st.sidebar.header("Upload or Use Sample Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

sample_excel = DataInterface.to_excel(DataInterface.sample_data())
if sample_excel:
    st.sidebar.download_button(
        "📥 Download Sample Template",
        data=sample_excel,
        file_name="pricing_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Load data
df = DataInterface.load(uploaded_file)
st.write(f"Loaded **{len(df)} rows**")

# Fit model
try:
    beta, r_sq, model = ModelInterface.fit(df)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
st.divider()
col1, col2, col3 = st.columns(3)

col1.metric("Elasticity (β)", f"{beta:.3f}")
col2.metric("R²", f"{r_sq:.2%}")
col3.metric("Market Type", "Elastic" if abs(beta) > 1 else "Inelastic")

# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------
st.header("🕹️ What‑If Simulation")

avg_q = df["Quantity"].mean()
current_price = st.number_input("Current Price ($)", value=float(df["Price"].mean()), min_value=0.01)
unit_cost = st.number_input("Unit Cost ($)", value=float(df["Price"].min() * 0.5), min_value=0.0)
price_change = st.slider("Adjust Price (%)", -50, 100, 0, step=5)

new_price = current_price * (1 + price_change / 100)
predicted_q = ModelInterface.predict_quantity(avg_q, new_price / current_price, beta)

new_revenue = new_price * predicted_q
new_profit = (new_price - unit_cost) * predicted_q

st.write(f"**Predicted Revenue:** ${new_revenue:,.2f}")
st.write(f"**Predicted Profit:** ${new_profit:,.2f}")

# Plot
prices = np.linspace(df["Price"].min() * 0.5, df["Price"].max() * 1.5, 60)
quantities = ModelInterface.predict_quantity(avg_q, prices / current_price, beta)
revenues = prices * quantities
profits = (prices - unit_cost) * quantities

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue"))
fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit"))
fig.add_vline(x=new_price, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)

# Diagnostics
with st.expander("📊 Model Diagnostics"):
    st.plotly_chart(
        px.scatter(df, x="Price", y="Quantity", trendline="ols"),
        use_container_width=True,
    )
    st.write(model.summary())

# Raw Data
with st.expander("📁 Raw Data"):
    st.dataframe(df)
