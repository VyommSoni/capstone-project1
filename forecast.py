import pandas as pd
from prophet import Prophet
from pulp import LpMaximize, LpProblem, LpVariable, lpSum


# Configuration Variables
BUDGET = 9000
ALPHA = 1.0  # Profit weight
BETA_SCALE = 100  # Scaling factor for beta
DEFAULT_MAX_STOCK = 20
DEFAULT_MIN_PURCHASE_UNITS = 1
MIN_HISTORY_FOR_FORECAST = 5  # Minimum sales history for forecasting
FORECAST_PERIOD = 30  # Days for forecast
default_forecasted_demand = 2 # minimal demand if insufficient history

# Load Data
sales_df = pd.read_csv("sales_data.csv")
product_df = pd.read_csv("top_products_for_optimization.csv")

# Required columns validation
required_sales_columns = {"product_id", "sale_date", "quantity", "current_stock","max_purchase_units", "min_purchase_units"}
if not required_sales_columns.issubset(set(sales_df.columns)):
    raise ValueError(f"sales_data.csv must contain columns: {required_sales_columns}")

# Merge stock, shelf space, and min purchase requirement
latest_stock_info = sales_df.drop_duplicates("product_id", keep="last")[["product_id", "current_stock", "min_purchase_units"]]
product_df = product_df.merge(latest_stock_info, on="product_id", how="left")
product_df = product_df.rename(columns={"current_stock": "max_stock"})


# Forecast future demand using Prophet
forecasted_demands = []

for product_id in product_df["product_id"]:
    df_product = sales_df[sales_df["product_id"] == product_id][["sale_date", "quantity"]]
    if len(df_product) < MIN_HISTORY_FOR_FORECAST:
        forecasted_demand = default_forecasted_demand
    else:
        df_product = df_product.rename(columns={"sale_date": "ds", "quantity": "y"})
        model = Prophet(daily_seasonality=True)
        model.fit(df_product)
        future = model.make_future_dataframe(periods=FORECAST_PERIOD)
        forecast = model.predict(future)
        forecasted_demand = forecast[-FORECAST_PERIOD:]["yhat"].clip(lower=0).sum()

    forecasted_demands.append(forecasted_demand)


product_df["forecasted_demand_30d"] = forecasted_demands
max_demand = product_df["forecasted_demand_30d"].max()
product_df["demand_score"] = product_df["forecasted_demand_30d"] / max_demand

# Adjusting weights dynamically for profit and demand
BETA = max(1.0, product_df["forecasted_demand_30d"].max() / BETA_SCALE)  # Scaled beta based on max demand
print(BETA)

# Adjust profit calculation based on forecasted values
product_df["adjusted_profit"] = product_df["avg_profit"] * product_df["forecasted_demand_30d"]

# Optimization Problem Setup
model = LpProblem("Retail_Optimization_With_Forecast", LpMaximize)

# Creating decision variables
variables = {
    row["product_id"]: LpVariable(f"qty_{row['product_id']}", lowBound=row["min_purchase_units"], upBound= min(row["max_stock"], row["avg_max_purchase_units"]), cat="Integer")
    for _, row in product_df.iterrows()
}

# Combined objective: profit + demand satisfaction
product_df["score"] = ALPHA * product_df["adjusted_profit"] + BETA * product_df["demand_score"]

# Maximize total score
model += lpSum(row["score"] * variables[row["product_id"]] for _, row in product_df.iterrows())

# Constraints: budget and shelf space
model += lpSum(row["avg_cost"] * variables[row["product_id"]] for _, row in product_df.iterrows()) <= BUDGET

# Solve optimization problem
model.solve()

# Results
total_spent = 0
total_profit = 0

print("\n\u2705 Recommended Purchase Plan:\n")
for _, row in product_df.iterrows():
    qty = int(variables[row["product_id"]].value())
    if qty > 0:
        cost = row["avg_cost"] * qty
        profit = row["avg_profit"] * qty
        total_spent += cost
        total_profit += profit
        print(f" {row['product_id']}: Buy {qty} units | Cost: ₹{cost:.0f} | Cost per unit: ₹{cost//qty:.0f} | Profit per unit : ₹{profit//qty:.0f} | Profit: ₹{profit:.0f}")

print(f"\n💵 Total Budget Used: ₹{total_spent:.0f}")
print(f"📈 Total Expected Profit: ₹{total_profit:.0f}")
