"""
Step 1 — Data Generation & Feature Engineering
Run: python src/data_prep.py
"""
import pandas as pd
import numpy as np
import os

def generate_dataset(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    categories     = ["Electronics","Fashion","Home & Kitchen","Sports","Beauty","Books","Toys","Food & Beverages"]
    cat_weights    = [0.20, 0.18, 0.15, 0.12, 0.12, 0.08, 0.08, 0.07]
    cat_base_price = {"Electronics":350,"Fashion":80,"Home & Kitchen":120,"Sports":95,
                      "Beauty":45,"Books":25,"Toys":40,"Food & Beverages":20}
    cat_elasticity = {"Electronics":-1.8,"Fashion":-2.1,"Home & Kitchen":-1.2,"Sports":-1.5,
                      "Beauty":-0.8,"Books":-0.6,"Toys":-1.9,"Food & Beverages":-0.5}

    cats       = np.random.choice(categories, size=n, p=cat_weights)
    base_p     = np.array([cat_base_price[c] for c in cats])
    prices     = np.round(base_p * np.random.uniform(0.7, 1.4, n), 2)
    comp_prices= np.round(prices * np.random.uniform(0.85, 1.15, n), 2)
    stock      = np.random.randint(0, 500, n)
    reviews    = np.round(np.random.uniform(2.5, 5.0, n), 1)
    days_on    = np.random.randint(1, 730, n)
    dow        = np.random.randint(0, 7, n)

    elasts     = np.array([cat_elasticity[c] for c in cats])
    log_demand = 4.5 + np.random.normal(0,0.3,n) + elasts*np.log(prices/base_p) + 0.3*(reviews-3.5)
    qty        = np.clip(np.round(np.exp(log_demand)).astype(int), 0, 500)

    cost_ratio = np.random.uniform(0.45, 0.70, n)
    unit_cost  = np.round(prices * cost_ratio, 2)
    margin_pct = np.round((prices - unit_cost) / prices * 100, 2)

    df = pd.DataFrame({
        "order_id":   [f"ORD-{str(i).zfill(6)}" for i in range(n)],
        "product_id": [f"PROD-{np.random.randint(1000,9999)}" for _ in range(n)],
        "date":       pd.date_range("2024-01-01", periods=n, freq="1h")[:n],
        "category":   cats,
        "price":      prices,
        "competitor_price": comp_prices,
        "unit_cost":  unit_cost,
        "quantity_sold": qty,
        "stock_level": stock,
        "review_score": reviews,
        "days_on_platform": days_on,
        "day_of_week": dow,
        "margin_pct": margin_pct,
        "revenue":    np.round(prices * qty, 2),
        "profit":     np.round((prices - unit_cost) * qty, 2),
    })
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/raw/ecommerce_orders.csv", index=False)
    print(f"Dataset saved → data/raw/ecommerce_orders.csv")
    print(f"Shape: {df.shape} | Revenue: €{df['revenue'].sum():,.0f} | Avg margin: {df['margin_pct'].mean():.1f}%")
