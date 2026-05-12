"""
Kyrgyzstan KIHS Dashboard — Final Project
Real data from the National Statistical Committee of the Kyrgyz Republic
(Нацстатком КР, https://stat.gov.kg)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kyrgyzstan KIHS Dashboard",
    page_icon="🇰🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

KGZ_RED  = "#E8112D"
KGZ_GOLD = "#FFEF00"
REGION_ORDER = ["Bishkek", "Osh city", "Chuy", "Issyk-Kul",
                "Naryn", "Talas", "Jalal-Abad", "Batken", "Osh"]

st.markdown("""
<style>
.hero{
  background: linear-gradient(135deg, #E8112D, #c0102a 60%, #8b0a1e);
  color: white; padding: 26px 32px; border-radius: 12px; margin-bottom: 22px;
}
.hero h1{margin:0; font-size: 2.0rem;}
.hero p {margin: 4px 0 0; opacity:.92;}
[data-testid="metric-container"]{
  background:white; border:1px solid #e6e6e6; border-radius:10px;
  padding:14px 16px; box-shadow:0 2px 5px rgba(0,0,0,.06);
}
.insight{ background:#fff8f0; border-left:4px solid #E8112D;
  padding:14px 18px; border-radius:8px; margin:8px 0; line-height:1.55;}
.insight.good { background:#f0fff4; border-color:#2e7d32;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NSC КР data…")
def load_data():
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "data", "nsc_kyrgyzstan_clean.csv"),
        os.path.join(base, "data", "nsc_kyrgyzstan_clean.csv"),
        "data/nsc_kyrgyzstan_clean.csv",
    ]
    nat_candidates = [
        os.path.join(base, "..", "data", "nsc_kyrgyzstan_national.csv"),
        "data/nsc_kyrgyzstan_national.csv",
    ]
    df = nat = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p); break
    for p in nat_candidates:
        if os.path.exists(p):
            nat = pd.read_csv(p); break
    if df is None or nat is None:
        st.error("Run the analysis notebook first to produce data/nsc_kyrgyzstan_clean.csv "
                 "and data/nsc_kyrgyzstan_national.csv "
                 "(or `python data/fetch_nsc_data.py` then the notebook).")
        st.stop()
    df["region"] = pd.Categorical(df["region"], REGION_ORDER, ordered=False)
    return df, nat


df, nat = load_data()


# ── Sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.caption("Real data — NSC КР · stat.gov.kg")
    st.markdown("---")

    years_all = sorted(df["year"].unique())
    yr_range = st.select_slider("📅 Year range",
                                options=years_all,
                                value=(min(years_all), max(years_all)))

    sel_regions = st.multiselect("🗺️ Oblast", REGION_ORDER, default=REGION_ORDER)

    indicator_map = {
        "Income (som/mo)":          "avg_per_capita_income_som",
        "Expenditure (som/mo)":     "avg_per_capita_expenditure_som",
        "Poverty rate (%)":         "poverty_rate_pct",
        "Poor population (thous.)": "poor_population_thousands",
        "Saving rate (%)":          "saving_rate_pct",
        "Real income YoY (%)":      "real_cash_income_yoy_pct",
        "Income vs national (%)":   "income_vs_national_pct",
        "Living wage (som)":        "living_wage_som",
        "Safe water access (%)":    "safe_drinking_water_share_pct",
        "Sewerage access (%)":      "sewerage_access_share_pct",
        "Electricity access (%)":   "electricity_access_share_pct",
    }
    indicator_label = st.selectbox("📊 Focus indicator",
                                   list(indicator_map.keys()), index=0)
    indicator = indicator_map[indicator_label]

    search = st.text_input("🔍 Search oblast")

    st.markdown("---")
    st.caption("Kyrgyzstan KIHS · Final Project")


# ── Apply filters ────────────────────────────────────────────────────────────
mask = (df["year"].between(yr_range[0], yr_range[1])
        & df["region"].isin(sel_regions))
if search:
    mask &= df["region"].astype(str).str.contains(search, case=False, na=False)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🇰🇬 Kyrgyzstan KIHS Dashboard</h1>
  <p>Household income, expenditure and poverty across the 9 oblasts ·
     Source: <b>National Statistical Committee of the Kyrgyz Republic</b> (stat.gov.kg)</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Showing **{len(fdf)}** of **{len(df)}** region-year observations · "
    f"data range {yr_range[0]}–{yr_range[1]}"
)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ov, tab_trend, tab_reg, tab_poverty, tab_explore = st.tabs([
    "📊 Overview",
    "📈 National Trends",
    "🗺️ Regional Comparison",
    "🏚️ Poverty Analysis",
    "🔍 Data Explorer & Insights",
])


# ═════════ TAB 1 — OVERVIEW (KPIs + key charts) ══════════════════════════════
with tab_ov:
    snap_year = int(fdf.dropna(subset=["avg_per_capita_income_som",
                                       "poverty_rate_pct"])["year"].max())
    snap = fdf[fdf["year"] == snap_year]

    st.markdown(f"### 🎯 Key indicators — reference year **{snap_year}**")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Avg income",
              f"{snap['avg_per_capita_income_som'].mean():,.0f} som")
    c2.metric("Avg expenditure",
              f"{snap['avg_per_capita_expenditure_som'].mean():,.0f} som")
    c3.metric("Mean poverty",
              f"{snap['poverty_rate_pct'].mean():.1f} %")
    c4.metric("Poor pop (Σ)",
              f"{snap['poor_population_thousands'].dropna().sum():,.0f} k")
    rich = snap.loc[snap["avg_per_capita_income_som"].idxmax(), "region"]
    poor = snap.loc[snap["poverty_rate_pct"].idxmax(), "region"]
    c5.metric("Richest oblast",  str(rich))
    c6.metric("Poorest oblast",  str(poor))

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        # Bar — income by region
        s = (snap.dropna(subset=["avg_per_capita_income_som"])
                  .sort_values("avg_per_capita_income_som", ascending=True))
        fig = px.bar(s, x="avg_per_capita_income_som", y=s["region"].astype(str),
                     orientation="h",
                     color="avg_per_capita_income_som",
                     color_continuous_scale="Reds",
                     text_auto=",.0f")
        fig.update_layout(title=f"Per-capita income by oblast — {snap_year}",
                          xaxis_title="Som / month", yaxis_title="",
                          height=380, margin=dict(t=40, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Pie / donut — share of poor population by oblast
        pop = snap.dropna(subset=["poor_population_thousands"]).copy()
        fig = px.pie(pop, names=pop["region"].astype(str),
                     values="poor_population_thousands", hole=.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(title=f"Share of poor population by oblast — {snap_year}",
                          height=380, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap — poverty by region × year
    st.markdown("### 🔥 Poverty rate heatmap (region × year)")
    pivot = (fdf.pivot_table(index="region", columns="year",
                             values="poverty_rate_pct", observed=True)
                .reindex(REGION_ORDER).dropna(how="all"))
    fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                    aspect="auto", text_auto=".0f",
                    labels=dict(color="Poverty %"))
    fig.update_layout(height=380, margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ═════════ TAB 2 — NATIONAL TRENDS ═══════════════════════════════════════════
with tab_trend:
    st.markdown("### 📈 Kyrgyz Republic — national-level trends")

    natf = nat[nat["year"].between(yr_range[0], yr_range[1])].sort_values("year")

    # Line — income vs expenditure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=natf["year"], y=natf["avg_per_capita_income_som"],
                             mode="lines+markers", name="Income",
                             line=dict(color=KGZ_RED, width=3),
                             fill="tozeroy",
                             fillcolor="rgba(232,17,45,0.08)"))
    fig.add_trace(go.Scatter(x=natf["year"], y=natf["avg_per_capita_expenditure_som"],
                             mode="lines+markers", name="Expenditure",
                             line=dict(color="#1f77b4", width=3, dash="dot")))
    fig.update_layout(title="Per-capita income vs expenditure — Kyrgyz Republic",
                      xaxis_title="Year", yaxis_title="Som / month / person",
                      height=440, margin=dict(t=50, b=0),
                      legend=dict(orientation="h", y=-.15))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Real YoY income growth
        nat_yoy = (natf[["year","real_cash_income_yoy_pct"]].dropna())
        fig = px.bar(nat_yoy, x="year", y="real_cash_income_yoy_pct",
                     color="real_cash_income_yoy_pct",
                     color_continuous_scale="RdYlGn", text_auto=".1f")
        fig.add_hline(y=100, line_dash="dash", line_color="black",
                      annotation_text="100 = no change", annotation_position="top right")
        fig.update_layout(title="Real cash income — YoY index",
                          xaxis_title="", yaxis_title="% of previous year",
                          height=380, margin=dict(t=50, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Living wage trend
        nat_lw = natf.dropna(subset=["living_wage_som"])
        if not nat_lw.empty:
            fig = px.line(nat_lw, x="year", y="living_wage_som",
                          markers=True, color_discrete_sequence=[KGZ_GOLD])
            fig.update_traces(line=dict(width=3))
            fig.update_layout(title="National minimum living wage",
                              xaxis_title="", yaxis_title="Som / month",
                              height=380, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ═════════ TAB 3 — REGIONAL COMPARISON ═══════════════════════════════════════
with tab_reg:
    st.markdown(f"### 🗺️ Regional comparison — **{indicator_label}**")

    # Time-series by region for chosen indicator
    fig = px.line(fdf.dropna(subset=[indicator]),
                  x="year", y=indicator,
                  color=fdf["region"].astype(str),
                  markers=True,
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(title=f"{indicator_label} by oblast",
                      xaxis_title="", yaxis_title=indicator_label,
                      height=480, margin=dict(t=50, b=0),
                      legend=dict(title="", orientation="v"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Box — distribution by region
        fig = px.box(fdf.dropna(subset=[indicator]),
                     x=fdf.dropna(subset=[indicator])["region"].astype(str),
                     y=indicator,
                     color=fdf.dropna(subset=[indicator])["region"].astype(str),
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     points="all")
        fig.update_layout(title=f"Spread of {indicator_label} by oblast",
                          xaxis_title="", yaxis_title=indicator_label,
                          showlegend=False,
                          height=420, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Latest-year ranking
        snap_y = int(fdf.dropna(subset=[indicator])["year"].max())
        snap2  = fdf[fdf["year"] == snap_y].dropna(subset=[indicator])
        snap2  = snap2.sort_values(indicator, ascending=True)
        fig = px.bar(snap2, x=indicator, y=snap2["region"].astype(str),
                     orientation="h",
                     color=indicator, color_continuous_scale="Reds",
                     text_auto=",.1f")
        fig.update_layout(title=f"{indicator_label} ranking — {snap_y}",
                          xaxis_title=indicator_label, yaxis_title="",
                          height=420, margin=dict(t=50, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🔗 Indicator correlation across regions × years")
    corr_cols = ["avg_per_capita_income_som","avg_per_capita_expenditure_som",
                 "poverty_rate_pct","saving_rate_pct",
                 "income_vs_national_pct","safe_drinking_water_share_pct",
                 "sewerage_access_share_pct","electricity_access_share_pct"]
    corr = fdf[corr_cols].corr().round(2)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=True, aspect="auto")
    fig.update_layout(height=440, margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ═════════ TAB 4 — POVERTY ═══════════════════════════════════════════════════
with tab_poverty:
    st.markdown("### 🏚️ Poverty: where it concentrates and how it moves")

    # Scatter — income vs poverty
    valid = fdf.dropna(subset=["avg_per_capita_income_som","poverty_rate_pct"])
    slope, intercept, r, p, _ = stats.linregress(
        valid["avg_per_capita_income_som"], valid["poverty_rate_pct"])

    fig = px.scatter(valid, x="avg_per_capita_income_som", y="poverty_rate_pct",
                     color=valid["region"].astype(str),
                     size="poor_population_thousands",
                     hover_data=["year"],
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     trendline="ols", trendline_scope="overall",
                     trendline_color_override="black")
    fig.update_layout(
        title=f"Income vs Poverty — Pearson r = {r:.2f}, p = {p:.1e}",
        xaxis_title="Per-capita income (som / month)",
        yaxis_title="Poverty rate (%)",
        height=520, margin=dict(t=50, b=0),
        legend=dict(title="Oblast"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Histogram — distribution of poverty rates
        fig = px.histogram(valid, x="poverty_rate_pct", nbins=15,
                           color_discrete_sequence=[KGZ_RED])
        fig.add_vline(x=valid["poverty_rate_pct"].mean(), line_dash="dash",
                      line_color="black",
                      annotation_text=f"mean = {valid['poverty_rate_pct'].mean():.1f}%")
        fig.update_layout(title="Distribution of regional poverty rates",
                          xaxis_title="Poverty rate (%)", yaxis_title="# observations",
                          height=380, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar — change in poverty over the window
        first_y, last_y = (int(valid["year"].min()),
                           int(valid["year"].max()))
        pv = (valid.pivot_table(index="region", columns="year",
                                values="poverty_rate_pct", observed=True)
                    .reindex(REGION_ORDER))
        if first_y in pv.columns and last_y in pv.columns:
            pv["delta"] = pv[last_y] - pv[first_y]
            pv = pv.dropna(subset=["delta"]).reset_index()
            pv["region"] = pv["region"].astype(str)
            pv = pv.sort_values("delta")
            fig = px.bar(pv, x="delta", y="region", orientation="h",
                         color="delta",
                         color_continuous_scale="RdYlGn_r",
                         text_auto=".1f")
            fig.update_layout(title=f"Δ poverty rate, {first_y} → {last_y}",
                              xaxis_title="pp change", yaxis_title="",
                              height=380, margin=dict(t=50, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ANOVA
    groups = [g["poverty_rate_pct"].dropna().values
              for _, g in fdf.groupby("region", observed=True)
              if g["poverty_rate_pct"].notna().any()]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        st.markdown(
            f"""<div class="insight">📐 <b>One-way ANOVA — does poverty differ between oblasts?</b><br>
            F = <b>{f_stat:.2f}</b>, p-value = <b>{p_val:.2e}</b>.
            {'The null hypothesis is rejected — regional poverty means are <b>not all equal</b>.' if p_val<0.05
             else 'No statistically significant difference at the 5% level.'}</div>""",
            unsafe_allow_html=True,
        )


# ═════════ TAB 5 — DATA EXPLORER & INSIGHTS ═════════════════════════════════
with tab_explore:
    st.markdown("### 💡 Three key insights")

    inc_premium = (fdf.groupby("region", observed=True)["income_vs_national_pct"]
                      .mean().sort_values(ascending=False))
    top_reg, top_pct = inc_premium.idxmax(), inc_premium.iloc[0]
    pov_avg = (fdf.groupby("region", observed=True)["poverty_rate_pct"]
                  .mean().sort_values(ascending=False))
    worst_reg, worst_pct = pov_avg.idxmax(), pov_avg.iloc[0]
    best_reg,  best_pct  = pov_avg.idxmin(), pov_avg.iloc[-1]

    valid = fdf.dropna(subset=["avg_per_capita_income_som","poverty_rate_pct"])
    _, _, r_glob, p_glob, _ = stats.linregress(
        valid["avg_per_capita_income_som"], valid["poverty_rate_pct"])

    st.markdown(
        f'<div class="insight"><b>1 · Bishkek\'s structural income premium.</b> '
        f'Across 2019–2023, <b>{top_reg}</b> sat at <b>{top_pct:.0f} %</b> '
        f'of the national average per-capita income — the gap with rural oblasts has not narrowed.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="insight"><b>2 · Naryn/Batken carry the poverty burden.</b> '
        f'Mean regional poverty 2019–2023: <b>{worst_reg}</b> = <b>{worst_pct:.1f} %</b>, '
        f'vs <b>{best_reg}</b> = <b>{best_pct:.1f} %</b> — a <b>{worst_pct/best_pct:.1f}×</b> gap.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="insight good"><b>3 · Income strongly anti-correlates with poverty.</b> '
        f'Across all (region × year) cells, Pearson r = <b>{r_glob:.2f}</b> '
        f'(p = {p_glob:.1e}). The income channel, not aid alone, drives the regional poverty map.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📋 Cleaned data explorer")
    st.dataframe(fdf, use_container_width=True, hide_index=True, height=380)

    st.download_button(
        "⬇️ Download cleaned CSV",
        data=fdf.to_csv(index=False).encode(),
        file_name="kihs_filtered.csv",
        mime="text/csv",
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🇰🇬 Kyrgyzstan KIHS Final Project · Data: National Statistical Committee "
    "of the Kyrgyz Republic (https://stat.gov.kg) · "
    "Licence: CC-BY-NC-SA 4.0"
)
