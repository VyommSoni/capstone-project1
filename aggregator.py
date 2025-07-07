from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, max as max_

# Step 1: Start Spark session
spark = SparkSession.builder \
    .appName("Retail Product Optimizer") \
    .getOrCreate()

# Step 2: Load your retail sales dataset
# Assumes the CSV has columns: product_id, sale_date, quantity, cost_price, selling_price, current_stock, max_stock, shelf_space_required, min_purchase_units
df = spark.read.csv("sales_data.csv", header=True, inferSchema=True)

# Step 3: Compute per-transaction profit
df = df.withColumn("profit", col("selling_price") - col("cost_price"))

# Step 4: Aggregate data per product
agg_df = df.groupBy("product_id").agg(
    avg("cost_price").alias("avg_cost"),
    avg("profit").alias("avg_profit"),
    sum("quantity").alias("total_units_sold"),
    count("*").alias("sales_frequency"),
    avg("current_stock").alias("avg_current_stock"),
    avg("max_stock").alias("avg_max_stock"),
    avg("min_purchase_units").alias("avg_min_purchase_units"), # Add min_purchase_units if provided in sales data
    avg("max_purchase_units").alias("avg_max_purchase_units")

)

# Step 5: Normalize demand score
# Get the maximum quantity sold for scaling
max_units_sold = agg_df.agg(max_("total_units_sold").alias("max_units")).collect()[0]["max_units"]
agg_df = agg_df.withColumn("demand_score", col("total_units_sold") / max_units_sold)

# Step 6: Compute Stock Gap (how much stock needs to be purchased)
agg_df = agg_df.withColumn("stock_gap", col("avg_max_stock") - col("avg_current_stock"))

# Step 7: Compute minimum units needed to buy
agg_df = agg_df.withColumn(
    "final_purchase_units",
    (col("avg_min_purchase_units") > col("stock_gap")).cast("int") * col("avg_min_purchase_units") +
    (col("avg_min_purchase_units") <= col("stock_gap")).cast("int") * col("stock_gap")
)

# Step 8: Optional filtering (e.g., ignore low-selling products or products with no stock gap)
filtered_products = agg_df.filter((col("total_units_sold") > 100) & (col("stock_gap") > 0))

# Step 9: Save to CSV for Python optimizer
filtered_products.select(
    "product_id",
    "avg_cost",
    "avg_profit",
    "demand_score",
    "stock_gap",
    "avg_max_purchase_units",
    "final_purchase_units"
).toPandas().to_csv("../data/top_products_for_optimization.csv", index=False)

# Stop Spark session
spark.stop()
