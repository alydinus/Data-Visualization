"""
Fetch real Kyrgyzstan socio-economic data from the National Statistical
Committee of the Kyrgyz Republic (Нацстатком КР) — https://stat.gov.kg/

Uses the open data JSON exports exposed at:
    https://stat.gov.kg/en/opendata/category/{ID}/json

License of source data: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

Output:
    data/nsc_kyrgyzstan_raw.csv   — long format (one row per region-year-indicator)
    data/nsc_kyrgyzstan_wide.csv  — wide format (one row per region-year)
"""

import json
import os
import time
import urllib.request

import pandas as pd

OUT_DIR    = os.path.dirname(__file__)
RAW_LONG   = os.path.join(OUT_DIR, "nsc_kyrgyzstan_raw.csv")
RAW_WIDE   = os.path.join(OUT_DIR, "nsc_kyrgyzstan_wide.csv")

BASE = "https://stat.gov.kg/en/opendata/category/{cat}/json"

# NSC category id → analytical column name.
# Mostly from the "Standard of living" branch (KIHS), complemented with a
# few regional indicators from neighbouring NSC branches (population,
# employment, healthcare, education, prices) so the resulting panel is rich
# enough for multi-angle analysis as required in 7chapter §04.04.
CATEGORIES = {
    # — Standard of living / KIHS core ────────────────────────────────────
    122:  "avg_per_capita_income_som",
    121:  "avg_per_capita_expenditure_som",
    120:  "poverty_rate_pct",
    295:  "poor_population_thousands",
    290:  "nominal_income_per_capita_som",
    291:  "real_cash_income_yoy_pct",
    119:  "living_wage_som",
    126:  "thermal_energy_expenditure_share_pct",
    125:  "energy_costs_share_pct",
    3104: "regional_income_to_national_pct",
    4296: "pension_recipients_thousands",
    4300: "old_age_pension_recipients_thousands",
    # — Infrastructure / access (KIHS context) ───────────────────────────
    5765: "safe_drinking_water_share_pct",
    5766: "electricity_access_share_pct",
    123:  "sewerage_access_share_pct",
}

# Categories whose primary axis is a demographic group rather than a region —
# kept on disk in the long-format file but excluded from the regional panel.
NON_REGIONAL_CATEGORIES: set[int] = set()

# Canonical region list for the analytical panel.
CANONICAL_REGIONS = {
    "Bishkek", "Osh city", "Chuy", "Issyk-Kul",
    "Naryn", "Talas", "Jalal-Abad", "Batken", "Osh",
    "Kyrgyz Republic",
}


def _fetch_one(cat_id: int) -> list[dict]:
    url = BASE.format(cat=cat_id)
    req = urllib.request.Request(url, headers={"User-Agent": "kihs-project/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("data", [])


def _normalize_region(name: str) -> str:
    """Map every variant NSC publishes to one of the CANONICAL_REGIONS labels.

    Demonstrates the kind of inconsistent text cleaning the 7chapter
    notebook §04.02 explicitly calls out as a required cleaning step.
    """
    n = name.strip()
    if "(" in n:                                  # strip parenthetical notes
        n = n.split("(")[0].strip()
    n_low = n.lower()
    fixes = {
        "kyrgyz republic":     "Kyrgyz Republic",
        "bishkek":             "Bishkek",
        "bishkek city":        "Bishkek",
        "osh":                 "Osh",
        "osh oblast":          "Osh",
        "osh city":            "Osh city",
        "chui":                "Chuy",
        "chuy":                "Chuy",
        "chui oblast":         "Chuy",
        "chuy oblast":         "Chuy",
        "yssyk-kul":           "Issyk-Kul",
        "yssyk-kul oblast":    "Issyk-Kul",
        "ysykkul":             "Issyk-Kul",
        "issyk-kul":           "Issyk-Kul",
        "issyk-kul oblast":    "Issyk-Kul",
        "jalal-abat":          "Jalal-Abad",
        "jalal-abad":          "Jalal-Abad",
        "djalal-abad":         "Jalal-Abad",
        "jalal-abat oblast":   "Jalal-Abad",
        "naryn":               "Naryn",
        "naryn oblast":        "Naryn",
        "talas":               "Talas",
        "talas oblast":        "Talas",
        "batken":              "Batken",
        "batken oblast":       "Batken",
    }
    return fixes.get(n_low, n)


def main():
    long_rows = []
    for cat_id, col in CATEGORIES.items():
        try:
            data = _fetch_one(cat_id)
            print(f"  cat {cat_id:>5}  → {col:<45} {len(data)} entities")
        except Exception as e:
            print(f"  cat {cat_id:>5}  ✗ {e}")
            continue

        for entry in data:
            raw_name = entry.get("title_en", "").strip()
            for v in entry.get("values", []):
                year = v.get("key")
                val  = v.get("value")
                if year is None or val is None:
                    continue
                long_rows.append({
                    "region_raw":        raw_name,         # keep messy strings — cleaning is part of the project
                    "region":            _normalize_region(raw_name),
                    "year":              int(year),
                    "indicator":         col,
                    "value":             float(val),
                    "source_category_id": cat_id,
                })
        time.sleep(0.4)  # be polite to the NSC server

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(RAW_LONG, index=False)
    print(f"\n✓ saved long-format raw → {RAW_LONG}  ({long_df.shape[0]} rows)")

    # Pivot to wide for the per-region-year panel used by the notebook/dashboard.
    panel = long_df[long_df["region"].isin(CANONICAL_REGIONS)].copy()
    wide = (panel
            .pivot_table(index=["region", "year"],
                         columns="indicator",
                         values="value",
                         aggfunc="first")
            .reset_index())
    wide.columns.name = None
    wide.to_csv(RAW_WIDE, index=False)
    print(f"✓ saved wide-format raw  → {RAW_WIDE}  ({wide.shape[0]} rows × {wide.shape[1]} cols)")
    print(f"  canonical regions present: {sorted(wide['region'].unique())}")
    print(f"  years:   {sorted(wide['year'].unique())}")
    print(f"  raw region strings observed (cleaning target): "
          f"{long_df['region_raw'].nunique()}")


if __name__ == "__main__":
    main()
