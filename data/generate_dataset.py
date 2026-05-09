"""
Generates a realistic multi-indicator dataset for Kyrgyzstan (2000-2023).
Sources for base values: World Bank, National Statistical Committee of the Kyrgyz Republic.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

YEARS = list(range(2000, 2024))
REGIONS = [
    "Bishkek", "Osh City", "Chuy", "Jalal-Abad",
    "Osh", "Batken", "Naryn", "Talas", "Issyk-Kul"
]

# --- National time-series data ---
gdp_per_capita = [
    280, 305, 330, 370, 430, 480, 530, 580, 700, 860,
    880, 1000, 1080, 1180, 1260, 1100, 1180, 1260, 1380, 1430,
    1680, 1610, 1900, 2080
]
population_millions = [
    4.9, 4.97, 5.03, 5.1, 5.17, 5.24, 5.32, 5.4, 5.48, 5.58,
    5.68, 5.78, 5.88, 5.99, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6,
    6.68, 6.76, 6.84, 6.93
]
inflation_rate = [
    18.7, 6.9, 2.1, 3.1, 4.1, 4.3, 5.6, 10.2, 24.5, 6.8,
    7.8, 16.5, 2.8, 6.6, 7.5, 6.5, 0.4, 3.2, 1.5, 3.1,
    6.3, 11.9, 14.7, 10.8
]
unemployment_rate = [
    7.5, 7.8, 7.5, 9.0, 8.5, 8.1, 8.3, 8.1, 8.0, 8.4,
    8.6, 8.5, 8.4, 8.1, 7.6, 7.5, 7.0, 6.9, 6.7, 6.6,
    5.9, 6.5, 5.9, 5.7
]
literacy_rate = [
    98.1, 98.2, 98.3, 98.4, 98.5, 98.6, 98.7, 98.8, 98.9, 99.0,
    99.1, 99.1, 99.2, 99.2, 99.3, 99.3, 99.4, 99.4, 99.5, 99.5,
    99.5, 99.5, 99.6, 99.6
]
life_expectancy = [
    65.2, 65.5, 65.8, 66.1, 66.4, 66.7, 67.0, 67.4, 67.7, 68.0,
    68.3, 68.6, 68.9, 69.2, 69.5, 69.7, 69.9, 70.1, 70.3, 70.5,
    70.6, 70.3, 70.8, 71.2
]
remittances_gdp_pct = [
    2.1, 2.5, 3.0, 5.0, 7.5, 15.0, 22.0, 27.5, 25.0, 20.0,
    22.0, 28.0, 30.7, 32.0, 31.5, 29.0, 30.4, 33.1, 29.7, 28.5,
    20.2, 32.4, 35.0, 33.5
]
exports_usd_bn = [
    0.51, 0.48, 0.49, 0.58, 0.73, 0.67, 0.81, 1.13, 1.84, 1.72,
    1.78, 2.27, 1.92, 2.08, 2.18, 1.87, 1.72, 1.80, 1.88, 1.99,
    2.30, 2.80, 3.30, 3.50
]
internet_users_pct = [
    1.0, 1.5, 2.2, 3.1, 4.0, 5.5, 8.0, 14.0, 20.0, 28.0,
    34.0, 40.0, 45.0, 49.0, 53.0, 58.0, 63.0, 68.0, 72.0, 75.0,
    78.0, 81.0, 83.0, 85.5
]
co2_emissions_kt = [
    6200, 5800, 5600, 6100, 6500, 7000, 7500, 7800, 9000, 8500,
    8200, 9100, 9300, 9500, 9700, 9400, 9200, 9600, 9800, 10200,
    9800, 9200, 10100, 10500
]

national_df = pd.DataFrame({
    "year": YEARS,
    "gdp_per_capita_usd": gdp_per_capita,
    "population_millions": population_millions,
    "inflation_rate_pct": inflation_rate,
    "unemployment_rate_pct": unemployment_rate,
    "literacy_rate_pct": literacy_rate,
    "life_expectancy_years": life_expectancy,
    "remittances_pct_gdp": remittances_gdp_pct,
    "exports_usd_bn": exports_usd_bn,
    "internet_users_pct": internet_users_pct,
    "co2_emissions_kt": co2_emissions_kt,
})

# Add noise to simulate real-world measurement variation
for col in ["inflation_rate_pct", "unemployment_rate_pct", "co2_emissions_kt"]:
    national_df[col] += np.random.normal(0, national_df[col].std() * 0.03, len(national_df))
    national_df[col] = national_df[col].round(2)

# Introduce a few realistic missing values
national_df.loc[national_df["year"].isin([2001, 2002]), "internet_users_pct"] = np.nan
national_df.loc[national_df["year"] == 2020, "exports_usd_bn"] = np.nan  # COVID disruption

# --- Regional data ---
REGION_POPULATION_2020 = {
    "Bishkek": 1074, "Osh City": 322, "Chuy": 898, "Jalal-Abad": 1137,
    "Osh": 1326, "Batken": 553, "Naryn": 280, "Talas": 253, "Issyk-Kul": 476
}
REGION_URBAN_PCT = {
    "Bishkek": 100, "Osh City": 100, "Chuy": 32, "Jalal-Abad": 22,
    "Osh": 18, "Batken": 15, "Naryn": 12, "Talas": 14, "Issyk-Kul": 28
}
REGION_POVERTY_PCT = {
    "Bishkek": 5.1, "Osh City": 12.3, "Chuy": 18.5, "Jalal-Abad": 35.2,
    "Osh": 38.7, "Batken": 41.2, "Naryn": 44.5, "Talas": 32.1, "Issyk-Kul": 25.8
}
REGION_SCHOOLS = {
    "Bishkek": 218, "Osh City": 76, "Chuy": 312, "Jalal-Abad": 495,
    "Osh": 601, "Batken": 241, "Naryn": 178, "Talas": 142, "Issyk-Kul": 198
}
REGION_HOSPITALS = {
    "Bishkek": 52, "Osh City": 18, "Chuy": 38, "Jalal-Abad": 44,
    "Osh": 48, "Batken": 22, "Naryn": 20, "Talas": 18, "Issyk-Kul": 26
}
REGION_AREA_KM2 = {
    "Bishkek": 127, "Osh City": 182, "Chuy": 20200, "Jalal-Abad": 33700,
    "Osh": 29200, "Batken": 17000, "Naryn": 45200, "Talas": 11400, "Issyk-Kul": 43100
}

regional_rows = []
for region in REGIONS:
    base_pop = REGION_POPULATION_2020[region]
    for i, year in enumerate(YEARS):
        growth = 1 + np.random.normal(0.012, 0.004)
        years_from_2020 = year - 2020
        pop = base_pop * (growth ** years_from_2020)
        gdp_index = gdp_per_capita[i]
        if region in ["Bishkek", "Osh City"]:
            regional_gdp = gdp_index * np.random.uniform(1.8, 2.3)
        elif region in ["Chuy", "Issyk-Kul"]:
            regional_gdp = gdp_index * np.random.uniform(1.1, 1.4)
        else:
            regional_gdp = gdp_index * np.random.uniform(0.5, 0.85)

        poverty = REGION_POVERTY_PCT[region] + np.random.normal(-0.3 * (i / len(YEARS)), 1.5)
        poverty = max(2.0, min(70.0, poverty))

        regional_rows.append({
            "year": year,
            "region": region,
            "population_thousands": round(pop, 1),
            "gdp_per_capita_usd": round(regional_gdp, 0),
            "poverty_rate_pct": round(poverty, 1),
            "urban_population_pct": REGION_URBAN_PCT[region] + np.random.normal(0, 0.5),
            "num_schools": REGION_SCHOOLS[region] + np.random.randint(-5, 6),
            "num_hospitals": REGION_HOSPITALS[region] + np.random.randint(-2, 3),
            "area_km2": REGION_AREA_KM2[region],
        })

regional_df = pd.DataFrame(regional_rows)
regional_df["population_density"] = (regional_df["population_thousands"] * 1000 / regional_df["area_km2"]).round(2)
regional_df["urban_population_pct"] = regional_df["urban_population_pct"].clip(0, 100).round(1)

# Introduce some missing values
regional_df.loc[regional_df.sample(frac=0.03, random_state=7).index, "poverty_rate_pct"] = np.nan
regional_df.loc[regional_df.sample(frac=0.02, random_state=11).index, "num_hospitals"] = np.nan

# Save raw datasets
national_df.to_csv("data/kyrgyzstan_national_indicators_raw.csv", index=False)
regional_df.to_csv("data/kyrgyzstan_regional_indicators_raw.csv", index=False)

print(f"National dataset: {national_df.shape}")
print(f"Regional dataset: {regional_df.shape}")
print("Raw datasets saved.")
