"""
Step 2 — Price Elasticity Estimation (log-log OLS regression per category)
Run: python src/elasticity_model.py
"""
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import os


def compute_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-log OLS: log(quantity) ~ log(price)
    Coefficient = price elasticity of demand per category.
    """
    results = []
    for cat in df["category"].unique():
        sub = df[(df["category"] == cat) & (df["quantity_sold"] > 0) & (df["price"] > 0)].copy()
        log_q = np.log(sub["quantity_sold"])
        log_p = np.log(sub["price"])
        model = OLS(log_q, add_constant(log_p)).fit()
        elast = model.params["price"]
        results.append({
            "category":        cat,
            "elasticity":      round(elast, 3),
            "r_squared":       round(model.rsquared, 3),
            "n_obs":           len(sub),
            "price_sensitivity": "Elastic (price-sensitive)" if abs(elast) > 1 else "Inelastic (price-insensitive)",
        })
    return pd.DataFrame(results).sort_values("elasticity")


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/ecommerce_orders.csv")
    elast_df = compute_elasticity(df)
    elast_df.to_csv("data/processed/price_elasticity.csv", index=False)
    print("Price elasticity saved → data/processed/price_elasticity.csv")
    print(elast_df.to_string(index=False))
