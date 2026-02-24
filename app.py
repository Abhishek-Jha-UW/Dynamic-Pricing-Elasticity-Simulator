# Complete Improved Code for Dynamic Pricing & Elasticity Simulator

import pandas as pd
import numpy as np
import json
from datetime import datetime

class DynamicPricingSimulator:
    def __init__(self, initial_price, demand_data):
        self.initial_price = initial_price
        self.demand_data = demand_data
        self.price_history = []
        self.demand_history = []
        self.validations = self.validate_data()

    def validate_data(self):
        if not isinstance(self.initial_price, (int, float)) or self.initial_price <= 0:
            raise ValueError("Initial price must be a positive number")
        if not isinstance(self.demand_data, pd.DataFrame):
            raise ValueError("Demand data must be a pandas DataFrame")
        return True

    def simulate_demand(self, price):
        return max(0, int(self.demand_data['base_demand'].mean() - price * self.demand_data['price_sensitivity'].mean()))

    def perform_pricing_strategy(self):
        current_price = self.initial_price
        for _ in range(10):  # Simulating for 10 price changes
            current_demand = self.simulate_demand(current_price)
            self.price_history.append(current_price)
            self.demand_history.append(current_demand)
            current_price *= 0.95  # Decrease price by 5%

    def report(self):
        report_data = {
            'date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'price_history': self.price_history,
            'demand_history': self.demand_history
        }
        return json.dumps(report_data, indent=4)

# Example usage:
if __name__ == '__main__':
    initial_price = 100.0
    demand_data = pd.DataFrame({
        'base_demand': [100, 150, 200],
        'price_sensitivity': [1.2, 1.5, 1.8]
    })
    simulator = DynamicPricingSimulator(initial_price, demand_data)
    simulator.perform_pricing_strategy()
    print(simulator.report())