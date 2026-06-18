"""
Step 4 & 5 — Rule-Based + XGBoost ML Pricing Engine
Run: python src/pricing_engine.py
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import os, warnings
warnings.filterwarnings("ignore")

from data_prep        import generate_dataset
from elasticity_model import compute_elasticity
from rfm_segmentation import rfm_segmentation


RULES = [
    (lambda r: r["stock_level"] > 300 and abs(r["elasticity"]) > 1.5,
     -0.08, "Overstock + high elasticity → reduce price to drive volume"),

    (lambda r: abs(r["elasticity"]) < 0.8 and r["margin_pct"] > 45,
     +0.06, "Inelastic demand + healthy margin → increase price"),

    (lambda r: r["competitor_price"] < r["price"] * 0.93 and r["segment"] in ["Loyal","Champions"],
     None,  "Competitor undercuts → match price to retain loyal customers"),

    (lambda r: r["stock_level"] < 50 and r["segment"] == "Champions",
     +0.04, "Low stock + champions segment → scarcity premium"),

    (lambda r: abs(r["elasticity"]) > 1.2 and r["competitor_price"] > r["price"] * 1.05,
     +0.03, "Competitor priced higher → opportunistic uptick"),
]

FEATURES = ["price","competitor_price","unit_cost","stock_level","review_score",
            "days_on_platform","day_of_week","elasticity","margin_pct",
            "RFM_score","cat_enc","seg_enc"]


def apply_rules(row) -> pd.Series:
    price = row["price"]
    for condition, adj, reason in RULES:
        if condition(row):
            if adj is None:
                adj = (row["competitor_price"] / price) - 1
            rec = round(price * (1 + adj), 2)
            new_margin = round((rec - row["unit_cost"]) / rec * 100, 2)
            return pd.Series([rec, round(adj*100,1), new_margin, reason])
    return pd.Series([price, 0.0, row["margin_pct"], "Hold current price"])


def train_ml_model(df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    y = df["recommended_price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"  MAE  : €{mean_absolute_error(y_te, y_pred):.2f}")
    print(f"  MAPE : {np.mean(np.abs((y_te-y_pred)/y_te))*100:.2f}%")
    print(f"  R²   : {r2_score(y_te, y_pred):.4f}")
    feat_imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    feat_imp.sort_values("importance", ascending=False).to_csv(
        "data/processed/feature_importance.csv", index=False)
    return model


def run_pipeline():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Step 1 — Generating dataset...")
    df = generate_dataset()
    df.to_csv("data/raw/ecommerce_orders.csv", index=False)

    print("Step 2 — Computing price elasticity...")
    elast = compute_elasticity(df)
    elast.to_csv("data/processed/price_elasticity.csv", index=False)

    print("Step 3 — RFM segmentation...")
    rfm = rfm_segmentation(df)
    rfm.to_csv("data/processed/rfm_segments.csv", index=False)

    print("Step 4 — Applying rule-based pricing engine...")
    df = df.merge(elast[["category","elasticity","price_sensitivity"]], on="category")
    df = df.merge(rfm[["product_id","segment","RFM_score"]], on="product_id", how="left")
    df["segment"] = df["segment"].fillna("Price-Sensitive")
    df[["recommended_price","price_change_pct","new_margin_pct","pricing_reason"]] = (
        df.apply(apply_rules, axis=1))
    repriced = (df["price_change_pct"] != 0).sum()
    print(f"  {repriced:,} / {len(df):,} SKUs repriced ({repriced/len(df)*100:.1f}%)")

    print("Step 5 — Training XGBoost ML pricing model...")
    le_cat = LabelEncoder(); le_seg = LabelEncoder()
    df["cat_enc"] = le_cat.fit_transform(df["category"])
    df["seg_enc"] = le_seg.fit_transform(df["segment"])
    model = train_ml_model(df)

    df["ml_recommended_price"] = model.predict(df[FEATURES].fillna(0)).round(2)
    df["ml_margin_pct"] = ((df["ml_recommended_price"] - df["unit_cost"])
                           / df["ml_recommended_price"] * 100).round(2)

    df.to_csv("data/processed/final_pricing_output.csv", index=False)
    print("\nPipeline complete → data/processed/final_pricing_output.csv")
    print(f"  Avg current margin : {df['margin_pct'].mean():.2f}%")
    print(f"  Avg ML margin      : {df['ml_margin_pct'].mean():.2f}%")


if __name__ == "__main__":
    run_pipeline()
