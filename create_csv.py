import pandas as pd
import random
from datetime import datetime, timedelta

# Define the number of records
num_records = 1000

# Generate random dates starting from 2024-01-01
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_records)]

# Generate random product IDs (assumes 10 products for simplicity)
product_ids = [f"Product_{random.randint(1, 10)}" for _ in range(num_records)]

# Generate random quantities sold
quantities = [random.randint(1, 15) for _ in range(num_records)]

cost_prices = [random.randint(10, 100) for _ in range(num_records)]

# Generate random selling prices (cost_price + random markup)
selling_prices = [cost + random.randint(12, 110) for cost in cost_prices]

# Define current stock and max stock values
# Random current stock (e.g., between 50 and 200 units) and max stock
current_stock = [random.randint(10, 50) for _ in range(num_records)]
max_stock = [random.randint(20, 60) for _ in range(num_records)]
min_purchase_units = [random.randint(1, 3) for _ in range(num_records)]
max_purchase_units = [random.randint(10, 30) for _ in range(num_records)]



# Create the DataFrame
sales_data = pd.DataFrame({
    'sale_date': dates,
    'product_id': product_ids,
    'quantity': quantities,
    'cost_price': cost_prices,
    'selling_price': selling_prices,
    'current_stock': current_stock,
    'max_stock': max_stock,
    'min_purchase_units' : min_purchase_units,
    'max_purchase_units' : max_purchase_units
})

# Save to CSV
sales_data.to_csv('../data/sales_data.csv', index=False,header=True)

print("sales_data.csv created successfully!")
