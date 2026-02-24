# Improved app.py

import sys

class CustomError(Exception):
    pass


def validate_input(data):
    if not data:
        raise CustomError("Input data cannot be empty")
    # Add further validation logic as necessary


def process_data(data):
    try:
        validate_input(data)
        # Further processing logic
        return f"Processed data: {data}"
    except CustomError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    data = input("Enter data: ")
    result = process_data(data)
    print(result)

if __name__ == '__main__':
    main()