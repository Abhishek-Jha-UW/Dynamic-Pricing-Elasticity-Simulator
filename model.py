import numpy as np
import pandas as pd
import statsmodels.api as sm

class PricingModel:
    def __init__(self, data):
        """
        Initializes with a dataframe containing 'Price' and 'Quantity'.
        """
        self.data = data[(data['Price'] > 0) & (data['Quantity'] > 0)].copy()
        self.model = None
        self.beta = None
        self.r_squared = None
        self.p_value = None

    def fit(self):
        """
        Performs Log-Log Regression to find Price Elasticity.
        ln(Q) = alpha + beta * ln(P)
        """
        log_p = np.log(self.data['Price'])
        log_q = np.log(self.data['Quantity'])
        
        X = sm.add_constant(log_p)
        self.model = sm.OLS(log_q, X).fit()
        
        # Beta is the coefficient of Price
        self.beta = self.model.params[1]
        self.r_squared = self.model.rsquared
        self.p_value = self.model.pvalues[1]
        
        return self.beta, self.r_squared

    def predict_quantity(self, current_price, current_quantity, target_price):
        """
        Predicts quantity at a new price point based on calculated elasticity.
        Formula: Q2 = Q1 * (P2 / P1) ^ beta
        """
        if self.beta is None:
            self.fit()
            
        quantity_ratio = (target_price / current_price) ** self.beta
        return current_quantity * quantity_ratio

    def get_optimal_price(self, unit_cost):
        """
        Calculates the price that maximizes profit.
        Formula derived from derivative of Profit = (P - Cost) * Q
        Only valid if beta < -1 (Elastic market).
        """
        if self.beta is None:
            self.fit()
            
        if self.beta >= -1:
            # If inelastic, the math suggests an infinite price; 
            # we return the max historical price as a proxy.
            return self.data['Price'].max()
        
        # Optimal Price formula for constant elasticity:
        optimal_price = (unit_cost * self.beta) / (1 + self.beta)
        return optimal_price
