"""
Streamlit Pricing KPI Dashboard
Run from project root: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pricing Analytics Dashboard",
                   layout="wide", page_icon="💰")


@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..")
    df       = pd.read_csv(f"{base}/data/processed/final_pricing_output.csv")
    elast    = pd.read_csv(f"{base}/data/processed/price_elasticity.csv")
    rfm      = pd.read_csv(f"{base}/data/processed/rfm_segments.csv")
    feat_imp = pd.read_csv(f"{base}/data/processed/feature_importance.csv")
    return df, elast, rfm, feat_imp

df, elast_df, rfm, feat_imp = load_data()

st.title("💰 E-Commerce Dynamic Pricing Dashboard")
st.markdown("**Pricing KPIs · Kundensegmentierung · Margin Optimierung**")
st.divider()

st.sidebar.header("🔽 Filters")
cat_filter = st.sidebar.multiselect("Category", df["category"].unique(), default=list(df["category"].unique()))
seg_filter = st.sidebar.multiselect("Customer Segment", df["segment"].unique(), default=list(df["segment"].unique()))
df_f = df[df["category"].isin(cat_filter) & df["segment"].isin(seg_filter)]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Total SKUs",        f"{len(df_f):,}")
c2.metric("🔄 SKUs Repriced",     f"{(df_f['price_change_pct'] != 0).sum():,}")
c3.metric("📊 Avg Current Margin", f"{df_f['margin_pct'].mean():.1f}%")
c4.metric("🤖 Avg ML Margin",     f"{df_f['ml_margin_pct'].mean():.1f}%")
c5.metric("💶 Total Revenue",     f"€{df_f['revenue'].sum()/1e6:.1f}M")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 Price Elasticity by Category")
    fig = px.bar(elast_df.sort_values("elasticity"),
                 x="elasticity", y="category", orientation="h",
                 color="price_sensitivity",
                 color_discrete_map={"Elastic (price-sensitive)": "#ff6b6b",
                                     "Inelastic (price-insensitive)": "#00d4aa"},
                 title="Elastic categories need volume-focused pricing")
    fig.add_vline(x=-1, line_dash="dash", line_color="#ffd166", annotation_text="-1 threshold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👥 Customer Segments (RFM)")
    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig2 = px.pie(seg_counts, names="Segment", values="Count",
                  color_discrete_sequence=["#6c63ff","#00d4aa","#ffd166"],
                  title="RFM-based customer segmentation")
    fig2.update_traces(textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("📈 Margin: Current vs ML Recommended")
    mc = df_f.groupby("category").agg(
        Current=("margin_pct","mean"),
        ML_Recommended=("ml_margin_pct","mean")).reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Current",        x=mc["category"], y=mc["Current"],        marker_color="#6c63ff"))
    fig3.add_trace(go.Bar(name="ML Recommended", x=mc["category"], y=mc["ML_Recommended"], marker_color="#00d4aa"))
    fig3.update_layout(barmode="group", yaxis_title="Avg Margin %",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("⚙️ Pricing Action Distribution")
    rc = df_f["pricing_reason"].value_counts().reset_index()
    rc.columns = ["Reason","Count"]
    rc["Reason"] = rc["Reason"].str[:52]
    fig4 = px.bar(rc, x="Count", y="Reason", orientation="h",
                  color_discrete_sequence=["#6c63ff"],
                  title=f"{(df_f['price_change_pct']!=0).sum():,} SKUs were repriced")
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("🔍 XGBoost Feature Importance")
fi = feat_imp.sort_values("importance").tail(8)
fig5 = px.bar(fi, x="importance", y="feature", orientation="h",
              color_discrete_sequence=["#ff6b6b"],
              title="Top drivers of ML price recommendation (R²=0.999)")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("📋 SKU-Level Pricing Recommendations")
cols = ["product_id","category","segment","price","competitor_price",
        "margin_pct","recommended_price","ml_recommended_price","pricing_reason"]
st.dataframe(
    df_f[cols].head(100).style.format({
        "price": "€{:.2f}", "competitor_price": "€{:.2f}",
        "recommended_price": "€{:.2f}", "ml_recommended_price": "€{:.2f}",
        "margin_pct": "{:.1f}%"}),
    use_container_width=True)

st.divider()
st.subheader("🤖 AI-Generated Pricing Report for Management")
if st.button("⚡ Generate Report"):
    top_cat   = df_f.groupby("category")["profit"].sum().idxmax()
    elastic   = elast_df[elast_df["price_sensitivity"].str.contains("Elastic")]["category"].tolist()
    inelastic = elast_df[~elast_df["price_sensitivity"].str.contains("Elastic")]["category"].tolist()
    repriced_pct = (df_f["price_change_pct"] != 0).sum() / len(df_f) * 100
    report = f"""
### Pricing Analytics Summary — {pd.Timestamp.now().strftime("%B %Y")}

Across **{len(df_f):,} analysed SKUs**, the engine recommends repricing **{repriced_pct:.1f}%** of the catalogue.

**🏆 Top Profit Category:** `{top_cat}`

**⚠️ High Price-Sensitivity:** `{', '.join(elastic)}` — avoid margin increases, focus on volume & competitor monitoring.

**✅ Margin Opportunity:** `{', '.join(inelastic)}` — inelastic demand, safe to apply +4–6% margin uplift.

**👥 Segment Actions:**
- **Champions**: Scarcity premium where stock < 50 units
- **Loyal**: Match competitor drops to protect retention
- **Price-Sensitive**: Competitive pricing over margin

*Generated by AI pricing engine | {pd.Timestamp.now().strftime("%d %b %Y")}*
"""
    st.markdown(report)
    st.success("Report ready to share with management.")
