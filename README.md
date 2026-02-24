# Dynamic-Pricing-Elasticity-Simulator

An interactive web application to calculate **Price Elasticity of Demand**. This tool features a real-time 'What-If' simulation engine designed to optimize revenue and profit margins.

---

### 1. What is Price Elasticity?
Price Elasticity of Demand (PED) measures how the quantity demanded of a good responds to a change in its price.

**Formula:**
$$PED = \frac{\% \text{ Change in Quantity Demanded}}{\% \text{ Change in Price}}$$

---

### 2. Log–Log Model (Constant Elasticity)
This model utilizes a **Log–Log regression** because the coefficients represent elasticities directly.

**Regression Equation:**
$$\ln(Q) = \beta_0 + \beta_1 \ln(P) + \epsilon$$

**Interpretation:**
*   A **1% increase** in Price ($P$) results in a **$\beta_1$% change** in Quantity ($Q$).
*   **Elasticity Rules:**
    *   $|\beta_1| > 1$: **Elastic** (High price sensitivity)
    *   $|\beta_1| < 1$: **Inelastic** (Low price sensitivity)

---

### 3. Features
*   **Automated Elasticity Engine:** Calculates elasticity using historical Price–Volume data. Includes statistical metrics like $R^2$ and p-values.
*   **What-If Simulator:** Provides revenue and profit optimization.
    *   **Optimal Profit Price:** $P_{opt} = \frac{Cost \times \beta}{1 + \beta}$
*   **Dynamic Visualizations:** Demand and revenue curves generated using **Plotly**.

---

### 4. Project Structure
*   `app.py`: Streamlit application interface.
*   `model.py`: Backend regression and optimization logic.
*   `requirements.txt`: Project dependencies.
*   `data/`: Directory for historical datasets.
