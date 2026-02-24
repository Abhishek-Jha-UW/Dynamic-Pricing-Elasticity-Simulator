# Improved app.py Code

import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        logging.info('Received data: %s', data)

        # Validate input
        if not data or 'price' not in data or 'quantity' not in data:
            logging.error('Invalid input data.')
            return jsonify({'error': 'Invalid input, please provide price and quantity.'}), 400

        price = data['price']
        quantity = data['quantity']

        # Perform calculations
        if price < 0 or quantity < 0:
            logging.error('Negative price or quantity found. Price: %s, Quantity: %s', price, quantity)
            return jsonify({'error': 'Price and quantity must be non-negative.'}), 400

        total_revenue = price * quantity
        logging.info('Calculated total revenue: %s', total_revenue)

        return jsonify({'total_revenue': total_revenue}), 200

    except Exception as e:
        logging.exception('An error occurred while calculating revenue: %s', e)
        return jsonify({'error': 'An error occurred while processing your request.'}), 500


if __name__ == '__main__':
    app.run(debug=True)