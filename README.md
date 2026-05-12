# Kyrgyzstan KIHS — Final Project

Data Analysis & Visualization course · Ala-Too International University.

End-to-end analytical pipeline on **real National Statistical Committee of the
Kyrgyz Republic** (Нацстатком КР) data covering household income, expenditure
and poverty across the 9 oblasts, 2018–2024.

## Layout

```
.
├── data/
│   ├── fetch_nsc_data.py             # downloads real NSC data from stat.gov.kg JSON API
│   ├── nsc_kyrgyzstan_raw.csv        # long-format raw (700 rows, region_raw kept messy)
│   ├── nsc_kyrgyzstan_wide.csv       # wide panel pivoted from raw (67 rows × 17 cols)
│   ├── nsc_kyrgyzstan_clean.csv      # cleaned regional panel produced by the notebook
│   └── nsc_kyrgyzstan_national.csv   # national time-series produced by the notebook
├── kihs_analysis.ipynb               # 12-section analysis notebook
├── dashboard/app.py                  # Streamlit interactive dashboard
├── viz_*.png                         # exported figures for the report
├── DEFENSE.md                        # full defense script (10–15 min)
├── 7chapter.ipynb                    # course capstone chapter (requirements ref)
├── Final Project.pdf                 # official rubric (requirements ref)
└── requirements.txt
```

## How to reproduce

```bash
# 1) install
pip install -r requirements.txt

# 2) fetch fresh data from NSC КР  (skip if data/*.csv already present)
python data/fetch_nsc_data.py

# 3) run the analysis notebook end-to-end
jupyter nbconvert --to notebook --execute kihs_analysis.ipynb \
                  --output kihs_analysis.ipynb

# 4) launch the dashboard
streamlit run dashboard/app.py
```

## Data source

National Statistical Committee of the Kyrgyz Republic — open data portal at
[https://stat.gov.kg/en/opendata/](https://stat.gov.kg/en/opendata/).
Licence: CC-BY-NC-SA 4.0.

See `DEFENSE.md` for the full project narrative and oral-defense flow.
