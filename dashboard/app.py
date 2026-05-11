"""Sales Analytics Dashboard — Final Project"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Technology":     "#1f77b4",
    "Furniture":      "#ff7f0e",
    "Office Supplies":"#2ca02c",
}
REGION_COLORS = px.colors.qualitative.Set1
SEG_COLORS    = px.colors.qualitative.Set2

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* gradient header */
  .dash-header {
    background: linear-gradient(135deg,#1565C0,#1976D2,#0097A7);
    padding:28px 32px; border-radius:12px; color:white; margin-bottom:24px;
  }
  .dash-header h1 { margin:0; font-size:2rem; font-weight:700; }
  .dash-header p  { margin:4px 0 0; font-size:.95rem; opacity:.85; }

  /* KPI cards */
  [data-testid="metric-container"] {
    background:white;
    border:1px solid #e0e0e0;
    border-radius:10px;
    padding:14px 18px;
    box-shadow:0 2px 6px rgba(0,0,0,.07);
  }

  /* insight boxes */
  .insight { background:#f0f7ff; border-left:4px solid #1565C0;
             padding:14px 18px; border-radius:8px; margin:8px 0;
             font-size:.92rem; line-height:1.6; }
  .insight.warn { background:#fff8e1; border-color:#f57c00; }

  /* subtle tab strip */
  button[data-baseweb="tab"] { font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading & cleaning ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    # locate CSV relative to this file or the project root
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "data", "sales_data.csv"),
        os.path.join(base, "data", "sales_data.csv"),
        "data/sales_data.csv",
    ]
    df = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        st.error("❌ `data/sales_data.csv` not found.  "
                 "Run `python data/generate_sales_data.py` first.")
        st.stop()

    # --- cleaning ---
    before = len(df)
    df.drop_duplicates(subset=["order_id"], inplace=True)
    df["discount"]       = df["discount"].fillna(df["discount"].median())
    df["shipping_cost"]  = df["shipping_cost"].fillna(df["shipping_cost"].median())

    # parse dates
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"]    = "Q" + df["order_date"].dt.quarter.astype(str)
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["day_of_week"]= df["order_date"].dt.strftime("%A")

    # outlier flag (3× IQR on sales)
    q1, q3 = df["sales"].quantile(.25), df["sales"].quantile(.75)
    iqr = q3 - q1
    df["is_outlier"] = (df["sales"] < q1 - 3*iqr) | (df["sales"] > q3 + 3*iqr)

    return df


df = load_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.markdown("---")

    min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
    date_range = st.date_input("📅 Date range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)

    sel_years    = st.multiselect("📆 Year",     sorted(df["year"].unique()),
                                  default=sorted(df["year"].unique()))
    sel_regions  = st.multiselect("🗺️ Region",   sorted(df["region"].unique()),
                                  default=sorted(df["region"].unique()))
    sel_cats     = st.multiselect("📦 Category", sorted(df["category"].unique()),
                                  default=sorted(df["category"].unique()))
    sel_segs     = st.multiselect("👥 Segment",  sorted(df["segment"].unique()),
                                  default=sorted(df["segment"].unique()))

    st.markdown("---")
    search = st.text_input("🔍 Search (product / city / customer)")

    st.markdown("---")
    st.caption("Sales Analytics Dashboard · Final Project")


# ── Filtering ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def apply_filters(df, date_range, years, regions, cats, segs, search):
    m = pd.Series([True] * len(df), index=df.index)
    if len(date_range) == 2:
        m &= df["order_date"].dt.date.between(date_range[0], date_range[1])
    if years:
        m &= df["year"].isin(years)
    if regions:
        m &= df["region"].isin(regions)
    if cats:
        m &= df["category"].isin(cats)
    if segs:
        m &= df["segment"].isin(segs)
    if search:
        s = search.lower()
        m &= (
            df["product_name"].str.lower().str.contains(s, na=False) |
            df["city"].str.lower().str.contains(s, na=False) |
            df["customer_name"].str.lower().str.contains(s, na=False) |
            df["sub_category"].str.lower().str.contains(s, na=False)
        )
    return df[m].copy()


fdf = apply_filters(df, date_range, sel_years, sel_regions, sel_cats, sel_segs, search)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 Sales Analytics Dashboard</h1>
  <p>Comprehensive Business Intelligence &amp; Performance Tracking · 2022–2024</p>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No records match the current filters. Please adjust the sidebar.")
    st.stop()

st.caption(f"Showing **{len(fdf):,}** of **{len(df):,}** orders")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_trend, tab_prod, tab_cust, tab_reg, tab_fc = st.tabs([
    "📊 Overview",
    "📈 Sales Trends",
    "🛍️ Products",
    "👥 Customers",
    "🗺️ Regional",
    "🔮 Forecast & Insights",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW / KPIs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_sales   = fdf["sales"].sum()
    total_profit  = fdf["profit"].sum()
    avg_revenue   = fdf["sales"].mean()
    cust_count    = fdf["customer_id"].nunique()
    total_orders  = len(fdf)
    margin_pct    = total_profit / total_sales * 100 if total_sales else 0

    best_sub = fdf.groupby("sub_category")["sales"].sum().idxmax()

    monthly_ts = fdf.groupby("year_month")["sales"].sum().sort_index()
    if len(monthly_ts) >= 2:
        mom = (monthly_ts.iloc[-1] - monthly_ts.iloc[-2]) / monthly_ts.iloc[-2] * 100
    else:
        mom = 0.0

    st.markdown("### 🎯 Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Total Sales",     f"${total_sales:,.0f}",  f"{mom:+.1f}% MoM")
    c2.metric("📈 Total Profit",    f"${total_profit:,.0f}", f"{margin_pct:.1f}% margin")
    c3.metric("💵 Avg Order Value", f"${avg_revenue:,.0f}")
    c4.metric("👥 Customers",       f"{cust_count:,}")
    c5.metric("📦 Orders",          f"{total_orders:,}")
    c6.metric("🏆 Best Product",    best_sub)

    st.markdown("---")

    # ── Two charts ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        cat_s = fdf.groupby("category")["sales"].sum().reset_index()
        fig = px.pie(cat_s, values="sales", names="category",
                     title="Sales by Category", hole=.42,
                     color="category", color_discrete_map=CAT_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=360, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        m_ts = fdf.groupby("year_month").agg(
            sales=("sales","sum"), profit=("profit","sum")
        ).reset_index().sort_values("year_month")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m_ts["year_month"], y=m_ts["sales"],
            mode="lines+markers", name="Sales",
            line=dict(color="#1976D2", width=2),
            fill="tozeroy", fillcolor="rgba(25,118,210,.1)"
        ))
        fig.add_trace(go.Scatter(
            x=m_ts["year_month"], y=m_ts["profit"],
            mode="lines+markers", name="Profit",
            line=dict(color="#2ca02c", width=2)
        ))
        fig.update_layout(
            title="Monthly Sales & Profit",
            xaxis_title="Month", yaxis_title="$",
            height=360, xaxis=dict(tickangle=-45),
            margin=dict(t=40,b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap: day × month ─────────────────────────────────────────────────
    st.markdown("### 🔥 Sales Heatmap — Day of Week × Month")
    day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    pivot = fdf.pivot_table(values="sales", index="day_of_week",
                            columns="month_name", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([d for d in day_order   if d in pivot.index])
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    title="Total Sales ($) by Day & Month",
                    text_auto=".2s", aspect="auto")
    fig.update_layout(height=320, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Segment bar ───────────────────────────────────────────────────────────
    seg_bar = fdf.groupby("segment").agg(
        Sales=("sales","sum"), Profit=("profit","sum"),
        Orders=("order_id","count")
    ).reset_index()

    fig = px.bar(seg_bar, x="segment", y=["Sales","Profit"],
                 barmode="group", title="Sales & Profit by Segment",
                 color_discrete_sequence=["#1976D2","#2ca02c"],
                 text_auto=".2s")
    fig.update_layout(height=340, xaxis_title="", yaxis_title="$",
                      margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SALES TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trend:
    st.markdown("### 📈 Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Quarterly grouped bar
        q_df = fdf.groupby(["year","quarter"]).agg(sales=("sales","sum")).reset_index()
        q_df["period"] = q_df["year"].astype(str) + " " + q_df["quarter"]
        fig = px.bar(q_df, x="period", y="sales", color="quarter",
                     title="Quarterly Sales",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     text_auto=".2s")
        fig.update_layout(height=380, xaxis_title="", yaxis_title="Sales ($)",
                          margin=dict(t=40,b=60), xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # YoY monthly comparison
        yoy = fdf.groupby(["year","month"])["sales"].sum().reset_index()
        fig = px.line(yoy, x="month", y="sales", color="year",
                      title="Year-over-Year Monthly Comparison",
                      markers=True,
                      color_discrete_sequence=REGION_COLORS)
        fig.update_layout(
            height=380,
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1,13)),
                ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
            ), margin=dict(t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stacked area: category over time
    cat_m = fdf.groupby(["year_month","category"])["sales"].sum().reset_index()
    cat_m.sort_values("year_month", inplace=True)
    fig = px.area(cat_m, x="year_month", y="sales", color="category",
                  title="Sales by Category Over Time (stacked)",
                  color_discrete_map=CAT_COLORS)
    fig.update_layout(height=360, xaxis=dict(tickangle=-45),
                      margin=dict(t=40,b=60))
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Sales vs Profit
    st.markdown("### 🔵 Sales vs Profit Scatter")
    sample = fdf.sample(min(2000, len(fdf)), random_state=42)
    fig = px.scatter(sample, x="sales", y="profit",
                     color="category", size="quantity",
                     hover_data=["customer_name","sub_category","region"],
                     title="Sales vs Profit (bubble = quantity)",
                     color_discrete_map=CAT_COLORS,
                     trendline="ols")
    fig.update_layout(height=460, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Discount vs avg profit
        disc = fdf.groupby("discount").agg(
            avg_profit=("profit","mean"), count=("order_id","count")
        ).reset_index()
        disc = disc[disc["count"] > 20]
        fig = px.bar(disc, x="discount", y="avg_profit",
                     title="Average Profit by Discount Level",
                     color="avg_profit", color_continuous_scale="RdYlGn",
                     text_auto=".0f")
        fig.update_layout(height=340, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Ship mode bar
        ship = fdf.groupby("ship_mode").agg(
            total_sales=("sales","sum"),
            avg_shipping=("shipping_cost","mean"),
        ).reset_index()
        fig = px.bar(ship, x="ship_mode", y="total_sales",
                     title="Sales by Ship Mode",
                     color="ship_mode",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     text_auto=".2s")
        fig.update_layout(height=340, showlegend=False,
                          xaxis_title="", margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prod:
    st.markdown("### 🛍️ Product Performance")

    top_n = st.slider("Show top N sub-categories", 5, 25, 10, key="top_n")

    prod = fdf.groupby(["category","sub_category"]).agg(
        total_sales=("sales","sum"),
        total_profit=("profit","sum"),
        total_qty=("quantity","sum"),
        orders=("order_id","count"),
        avg_discount=("discount","mean"),
    ).reset_index().sort_values("total_sales", ascending=False)

    top = prod.head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(top.sort_values("total_sales"),
                     x="total_sales", y="sub_category",
                     color="category", orientation="h",
                     title=f"Top {top_n} Sub-Categories by Sales",
                     color_discrete_map=CAT_COLORS, text_auto=".2s")
        fig.update_layout(height=420, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(top.sort_values("total_profit"),
                     x="total_profit", y="sub_category",
                     color="category", orientation="h",
                     title=f"Top {top_n} Sub-Categories by Profit",
                     color_discrete_map=CAT_COLORS, text_auto=".2s")
        fig.update_layout(height=420, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Bubble: Sales vs Profit vs Qty
    sub_agg = fdf.groupby(["sub_category","category"]).agg(
        sales=("sales","sum"), profit=("profit","sum"), qty=("quantity","sum")
    ).reset_index()
    fig = px.scatter(sub_agg, x="sales", y="profit",
                     size="qty", color="category", hover_name="sub_category",
                     title="Product Bubble Chart (bubble = total units sold)",
                     color_discrete_map=CAT_COLORS, size_max=55)
    fig.update_layout(height=460, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Category summary table
    st.markdown("### 📊 Category Summary Table")
    cat_tbl = fdf.groupby("category").agg(
        Total_Sales=("sales","sum"),
        Total_Profit=("profit","sum"),
        Orders=("order_id","count"),
        Unique_Customers=("customer_id","nunique"),
        Avg_Discount_pct=("discount","mean"),
    ).reset_index()
    cat_tbl["Profit_Margin_%"] = (cat_tbl["Total_Profit"] / cat_tbl["Total_Sales"] * 100).round(1)
    cat_tbl["Avg_Discount_%"]  = (cat_tbl["Avg_Discount_pct"] * 100).round(1)
    cat_tbl["Total_Sales"]     = cat_tbl["Total_Sales"].map("${:,.0f}".format)
    cat_tbl["Total_Profit"]    = cat_tbl["Total_Profit"].map("${:,.0f}".format)
    st.dataframe(cat_tbl.drop(columns="Avg_Discount_pct"),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cust:
    st.markdown("### 👥 Customer Analysis")

    col1, col2 = st.columns(2)

    with col1:
        seg_s = fdf.groupby("segment")["sales"].sum().reset_index()
        fig = px.pie(seg_s, values="sales", names="segment",
                     title="Sales by Customer Segment",
                     color_discrete_sequence=SEG_COLORS, hole=.38)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=360, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_c = (
            fdf.groupby(["customer_id","customer_name"])
               .agg(total_sales=("sales","sum"), orders=("order_id","count"))
               .reset_index()
               .sort_values("total_sales", ascending=False)
               .head(15)
        )
        fig = px.bar(top_c.sort_values("total_sales"),
                     x="total_sales", y="customer_name", orientation="h",
                     title="Top 15 Customers by Sales",
                     color="total_sales", color_continuous_scale="Blues",
                     text_auto=".2s")
        fig.update_layout(height=420, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Segment × Category heatmap
    st.markdown("### 🔥 Segment × Category Heatmap")
    seg_cat = fdf.pivot_table(values="sales", index="segment",
                               columns="category", aggfunc="sum", fill_value=0)
    fig = px.imshow(seg_cat, title="Sales: Segment × Category",
                    color_continuous_scale="Blues", text_auto=".2s", aspect="auto")
    fig.update_layout(height=280, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Purchase frequency histogram
        freq = fdf.groupby("customer_id")["order_id"].count().reset_index()
        freq.columns = ["customer_id","orders"]
        fig = px.histogram(freq, x="orders", nbins=20,
                           title="Customer Purchase Frequency",
                           color_discrete_sequence=["#1976D2"])
        fig.update_layout(height=340, xaxis_title="Orders per Customer",
                          yaxis_title="# Customers", margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Segment trends over time
        seg_m = fdf.groupby(["year_month","segment"])["sales"].sum().reset_index()
        fig = px.line(seg_m.sort_values("year_month"),
                      x="year_month", y="sales", color="segment",
                      title="Segment Sales Trends",
                      color_discrete_sequence=SEG_COLORS, markers=False)
        fig.update_layout(height=340, xaxis=dict(tickangle=-45),
                          margin=dict(t=40,b=60))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REGIONAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("### 🗺️ Regional Performance")

    reg = fdf.groupby("region").agg(
        total_sales=("sales","sum"),
        total_profit=("profit","sum"),
        orders=("order_id","count"),
        customers=("customer_id","nunique"),
    ).reset_index()
    reg["profit_margin_%"] = (reg["total_profit"] / reg["total_sales"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(reg.sort_values("total_sales", ascending=False),
                     x="region", y="total_sales",
                     title="Total Sales by Region",
                     color="region", color_discrete_sequence=REGION_COLORS,
                     text_auto=".2s")
        fig.update_layout(height=380, showlegend=False,
                          xaxis_title="", margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(reg.sort_values("profit_margin_%", ascending=False),
                     x="region", y="profit_margin_%",
                     title="Profit Margin by Region (%)",
                     color="profit_margin_%", color_continuous_scale="RdYlGn",
                     text_auto=".1f")
        fig.update_layout(height=380, xaxis_title="", margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Top cities
    st.markdown("### 🏙️ Top Cities")
    n_cities = st.slider("Show top N cities", 5, 25, 15, key="n_cities")
    city_df = (
        fdf.groupby(["region","city"])
           .agg(total_sales=("sales","sum"), total_profit=("profit","sum"),
                orders=("order_id","count"))
           .reset_index()
           .sort_values("total_sales", ascending=False)
           .head(n_cities)
    )
    fig = px.bar(city_df.sort_values("total_sales"),
                 x="total_sales", y="city", color="region",
                 orientation="h",
                 title=f"Top {n_cities} Cities by Sales",
                 color_discrete_sequence=REGION_COLORS, text_auto=".2s")
    fig.update_layout(height=420, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Region × Category heatmap
    st.markdown("### 🔥 Region × Category Heatmap")
    reg_cat = fdf.pivot_table(values="sales", index="region",
                               columns="category", aggfunc="sum", fill_value=0)
    fig = px.imshow(reg_cat, title="Sales: Region × Category",
                    color_continuous_scale="YlOrRd",
                    text_auto=".2s", aspect="auto")
    fig.update_layout(height=300, margin=dict(t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Regional trends
    reg_m = fdf.groupby(["year_month","region"])["sales"].sum().reset_index()
    fig = px.line(reg_m.sort_values("year_month"),
                  x="year_month", y="sales", color="region",
                  title="Regional Sales Trends Over Time",
                  color_discrete_sequence=REGION_COLORS)
    fig.update_layout(height=380, xaxis=dict(tickangle=-45),
                      margin=dict(t=40,b=60))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FORECAST & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fc:
    st.markdown("### 🔮 Sales Forecast — ML Predictive Analytics")

    # Use full dataset for forecast (not filtered) for better model stability
    m_agg = (
        df.groupby("year_month")["sales"].sum()
          .reset_index()
          .sort_values("year_month")
    )
    m_agg["t"] = np.arange(len(m_agg))

    if len(m_agg) >= 6:
        X = m_agg[["t"]].values
        y = m_agg["sales"].values

        poly = PolynomialFeatures(degree=2, include_bias=True)
        Xp   = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        y_fit = model.predict(Xp)
        r2    = r2_score(y, y_fit)

        # Forecast 6 months ahead
        last_t = m_agg["t"].max()
        future_t = np.arange(last_t + 1, last_t + 7).reshape(-1, 1)
        future_Xp = poly.transform(future_t)
        forecast  = model.predict(future_Xp)

        # Build future month labels
        last_period = pd.Period(m_agg["year_month"].iloc[-1], "M")
        future_labels = [(last_period + i).strftime("%Y-%m") for i in range(1, 7)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m_agg["year_month"], y=y,
            mode="lines+markers", name="Actual",
            line=dict(color="#1976D2", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=m_agg["year_month"], y=y_fit,
            mode="lines", name="Model fit",
            line=dict(color="#ff7f0e", width=1, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=future_labels, y=forecast,
            mode="lines+markers", name="6-Month Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(symbol="diamond", size=9)
        ))
        last_label = m_agg["year_month"].iloc[-1]
        fig.add_shape(type="line",
                      x0=last_label, x1=last_label,
                      y0=0, y1=1, yref="paper",
                      line=dict(color="gray", width=1.5, dash="dash"))
        fig.add_annotation(x=last_label, y=1, yref="paper",
                           text="Forecast →", showarrow=False,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=11))
        fig.update_layout(
            title=f"Monthly Sales Forecast — Polynomial Regression (R² = {r2:.3f})",
            xaxis_title="Month", yaxis_title="Sales ($)",
            height=500, xaxis=dict(tickangle=-45),
            margin=dict(t=50,b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

        fc_tbl = pd.DataFrame({
            "Month": future_labels,
            "Forecasted Sales": [f"${v:,.0f}" for v in forecast],
        })
        st.markdown("#### Forecast Values")
        st.dataframe(fc_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough monthly data to build a forecast. Adjust filters.")

    # ── Business Insights ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Key Business Insights")

    # 1. Top revenue products
    top3_sub = fdf.groupby("sub_category")["sales"].sum().sort_values(ascending=False).head(3)
    top3_str  = ", ".join(f"<b>{p}</b> (${v:,.0f})" for p, v in top3_sub.items())
    st.markdown(f'<div class="insight">🏆 <b>Top Revenue Products:</b> {top3_str}</div>',
                unsafe_allow_html=True)

    # 2. Peak month
    mo = fdf.groupby("month_name")["sales"].sum()
    mo = mo.reindex([m for m in ["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"] if m in mo.index])
    peak_m, low_m = mo.idxmax(), mo.idxmin()
    st.markdown(
        f'<div class="insight">📅 <b>Seasonality:</b> Peak month = <b>{peak_m}</b> '
        f'(${mo[peak_m]:,.0f}). Slowest = <b>{low_m}</b> (${mo[low_m]:,.0f}). '
        f'Q4 (Oct–Dec) consistently drives the highest sales volume.</div>',
        unsafe_allow_html=True
    )

    # 3. Best region
    top_reg = fdf.groupby("region")["sales"].sum().idxmax()
    top_reg_val = fdf.groupby("region")["sales"].sum().max()
    low_margin_reg = fdf.groupby("region").apply(
        lambda x: x["profit"].sum() / x["sales"].sum() if x["sales"].sum() > 0 else 0
    ).idxmin()
    st.markdown(
        f'<div class="insight">🗺️ <b>Regional Performance:</b> Best region = '
        f'<b>{top_reg}</b> (${top_reg_val:,.0f} total sales). '
        f'Region with lowest profit margin: <b>{low_margin_reg}</b> — '
        f'review pricing & discount policy there.</div>',
        unsafe_allow_html=True
    )

    # 4. Customer patterns
    repeat_rate = (fdf.groupby("customer_id")["order_id"].count() > 1).mean() * 100
    top_seg     = fdf.groupby("segment")["sales"].sum().idxmax()
    st.markdown(
        f'<div class="insight">👥 <b>Customer Patterns:</b> '
        f'<b>{repeat_rate:.1f}%</b> of customers placed more than one order. '
        f'<b>{top_seg}</b> segment drives the highest revenue. '
        f'Top 10% of customers account for a disproportionate share of total sales.</div>',
        unsafe_allow_html=True
    )

    # 5. Underperforming category
    cat_mg = fdf.groupby("category").apply(
        lambda x: x["profit"].sum() / x["sales"].sum() * 100 if x["sales"].sum() > 0 else 0
    ).sort_values()
    worst_cat, worst_mg = cat_mg.index[0], cat_mg.iloc[0]
    st.markdown(
        f'<div class="insight warn">⚠️ <b>Underperforming Category:</b> '
        f'<b>{worst_cat}</b> has the lowest profit margin at <b>{worst_mg:.1f}%</b>. '
        f'High discount rates are eroding profitability — '
        f'consider capping discounts at 20% for this category.</div>',
        unsafe_allow_html=True
    )

    # 6. Discount impact
    hi_disc = fdf[fdf["discount"] >= 0.30]["profit"].mean()
    lo_disc = fdf[fdf["discount"] <  0.10]["profit"].mean()
    st.markdown(
        f'<div class="insight warn">💸 <b>Discount Impact:</b> '
        f'Orders with ≥30% discount average <b>${hi_disc:,.0f}</b> profit, '
        f'vs <b>${lo_disc:,.0f}</b> for low/no-discount orders. '
        f'Excessive discounting significantly reduces margin.</div>',
        unsafe_allow_html=True
    )

    # ── Raw data explorer ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Data Explorer")
    cols_show = ["order_id","order_date","customer_name","segment","region",
                 "city","category","sub_category","quantity","unit_price",
                 "discount","sales","profit","ship_mode"]
    st.dataframe(fdf[cols_show].head(200), use_container_width=True, hide_index=True)

    csv_bytes = fdf.to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered Data (CSV)",
                       data=csv_bytes, file_name="sales_filtered.csv",
                       mime="text/csv")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Sales Analytics Dashboard · Data Analysis & Visualization · Final Project")
