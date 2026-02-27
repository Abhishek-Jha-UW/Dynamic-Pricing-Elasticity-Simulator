# ============================================================
# Price Elasticity & Profit Optimization App
# Professional Version – Interpretable & Business Friendly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from model import PricingModel

st.set_page_config(
    page_title="Price Elasticity & Profit Optimizer",
    layout="wide"
)

# ============================================================
# Header
# ============================================================

st.title("📊 Price Elasticity & Profit Optimization Tool")
st.markdown("""
Estimate price elasticity using log-log regression, simulate pricing scenarios,
and identify profit-maximizing price points.
""")

# ============================================================
# Template Download
# ============================================================

st.subheader("1️⃣ Download Data Template")

template_df = pd.DataFrame({
    "Price": [10, 12, 15, 18, 20],
    "Quantity": [500, 460, 400, 350, 300]
})

csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="⬇ Download CSV Template",
    data=csv_buffer.getvalue(),
    file_name="pricing_template.csv",
    mime="text/csv"
)

st.markdown("---")

# ============================================================
# Upload Section
# ============================================================

st.subheader("2️⃣ Upload Your Data")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        # ============================================================
        # Column Mapping (Prevents Header Errors)
        # ============================================================

        st.subheader("3️⃣ Map Your Columns")

        price_col = st.selectbox("Select Price Column", df.columns)
        quantity_col = st.selectbox("Select Quantity Column", df.columns)

        if price_col and quantity_col:

            model_data = df[[price_col, quantity_col]].rename(
                columns={price_col: "Price", quantity_col: "Quantity"}
            )

            # ============================================================
            # Model Fitting
            # ============================================================

            model = PricingModel(model_data)
            beta, r2 = model.fit()

            st.markdown("---")
            st.header("📈 Regression Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Elasticity (β)", round(beta, 3))
            col2.metric("R²", round(model.r_squared, 3))
            col3.metric("p-value", round(model.p_value, 4))

            # Interpretation
            if beta < -1:
                elasticity_type = "Elastic"
                interpretation = "Demand is highly responsive to price changes."
            elif -1 <= beta < 0:
                elasticity_type = "Inelastic"
                interpretation = "Demand is relatively insensitive to price changes."
            else:
                elasticity_type = "Unusual (Positive Elasticity)"
                interpretation = "Data suggests upward-sloping demand. Check dataset."

            st.info(f"""
            **Market Type:** {elasticity_type}  
            {interpretation}
            """)

            # ============================================================
            # Show Full Regression Output
            # ============================================================

            with st.expander("📄 View Full Regression Output"):
                st.text(model.model.summary())

            # ============================================================
            # Visualization
            # ============================================================

            st.header("📊 Demand Curve")

            fig, ax = plt.subplots()
            ax.scatter(model_data["Price"], model_data["Quantity"])
            ax.set_xlabel("Price")
            ax.set_ylabel("Quantity")
            ax.set_title("Observed Price vs Quantity")

            st.pyplot(fig)

            # ============================================================
            # What-If Simulation
            # ============================================================

            st.markdown("---")
            st.header("🔮 What-If Simulation")

            current_price = st.number_input("Current Price", value=float(model_data["Price"].mean()))
            current_quantity = st.number_input("Current Quantity", value=float(model_data["Quantity"].mean()))
            target_price = st.number_input("New Target Price")

            if st.button("Simulate Demand"):

                predicted_q = model.predict_quantity(
                    current_price,
                    current_quantity,
                    target_price
                )

                st.success(f"Predicted Quantity at ${target_price:.2f}: {predicted_q:.2f}")

                revenue_old = current_price * current_quantity
                revenue_new = target_price * predicted_q

                st.write(f"Old Revenue: ${revenue_old:,.2f}")
                st.write(f"New Revenue: ${revenue_new:,.2f}")

                change_pct = ((revenue_new - revenue_old) / revenue_old) * 100

                st.metric("Revenue Change %", f"{change_pct:.2f}%")

            # ============================================================
            # Profit Optimization
            # ============================================================

            st.markdown("---")
            st.header("💰 Profit Maximization")

            unit_cost = st.number_input("Enter Unit Cost")

            if st.button("Calculate Optimal Price"):

                optimal_price = model.get_optimal_price(unit_cost)

                st.success(f"Optimal Price: ${optimal_price:.2f}")

                st.info("""
                ⚠ Optimal price formula assumes:
                - Constant elasticity demand
                - Stable cost structure
                - No competitive reactions
                """)

            # ============================================================
            # Executive Summary
            # ============================================================

            st.markdown("---")
            st.header("📌 Executive Summary")

            st.write(f"""
            - Estimated Price Elasticity: **{round(beta,3)}**
            - Model Fit (R²): **{round(r2,3)}**
            - Market Type: **{elasticity_type}**
            - Recommendation: {'Price reductions can increase revenue.' if beta < -1 else 'Price increases may improve margins.'}
            """)

    except Exception as e:
        st.error(f"Error processing file: {e}")
