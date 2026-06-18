"""
Step 3 — RFM Customer / Product Segmentation
Run: python src/rfm_segmentation.py
"""
import pandas as pd
import os


def score_col(series, ascending=True):
    labels = [1, 2, 3, 4] if ascending else [4, 3, 2, 1]
    try:
        return pd.qcut(series, q=4, labels=labels, duplicates="drop").astype(int)
    except Exception:
        return pd.cut(series, bins=4, labels=labels, include_lowest=True).astype(int)


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    snapshot = df["date"].max()

    rfm = df.groupby("product_id").agg(
        recency=("date",    lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "count"),
        monetary=("revenue", "sum"),
    ).reset_index()

    rfm["R_score"] = score_col(rfm["recency"],   ascending=False)
    rfm["F_score"] = score_col(rfm["frequency"], ascending=True)
    rfm["M_score"] = score_col(rfm["monetary"],  ascending=True)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    rfm["segment"] = rfm["RFM_score"].apply(
        lambda s: "Champions" if s >= 10 else ("Loyal" if s >= 7 else "Price-Sensitive")
    )
    return rfm


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/ecommerce_orders.csv")
    rfm = rfm_segmentation(df)
    rfm.to_csv("data/processed/rfm_segments.csv", index=False)
    print("RFM segments saved → data/processed/rfm_segments.csv")
    print(rfm["segment"].value_counts().to_string())
