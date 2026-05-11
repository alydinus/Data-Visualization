import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

np.random.seed(42)
random.seed(42)

N = 10000

catalog = {
    'Technology': {
        'Laptops':      (700, 2500),
        'Smartphones':  (300, 1500),
        'Tablets':      (200, 1000),
        'Monitors':     (150,  800),
        'Keyboards':    ( 20,  200),
        'Webcams':      ( 30,  250),
        'Headphones':   ( 30,  500),
        'Printers':     (100,  600),
        'Hard Drives':  ( 50,  300),
        'USB Hubs':     ( 15,   80),
    },
    'Furniture': {
        'Office Chairs':  (100,  800),
        'Standing Desks': (300, 1500),
        'Bookshelves':    ( 80,  400),
        'Filing Cabinets':( 60,  350),
        'Sofas':          (400, 2000),
        'Coffee Tables':  (100,  600),
        'Wardrobes':      (200, 1200),
        'Bed Frames':     (150,  800),
    },
    'Office Supplies': {
        'Notebooks':        ( 5,  30),
        'Pens & Pencils':   ( 2,  25),
        'Paper Reams':      ( 8,  50),
        'Binders':          ( 5,  40),
        'Staplers':         (10,  60),
        'Sticky Notes':     ( 3,  20),
        'Desk Organizers':  (15,  80),
        'Scissors':         ( 5,  30),
        'Tape & Glue':      ( 3,  25),
        'Folders':          ( 5,  35),
    },
}

regions_cities = {
    'North':   ['Chicago', 'Detroit', 'Minneapolis', 'Milwaukee', 'Cleveland'],
    'South':   ['Houston', 'Dallas', 'Atlanta', 'Miami', 'Nashville'],
    'East':    ['New York', 'Philadelphia', 'Boston', 'Washington DC', 'Baltimore'],
    'West':    ['Los Angeles', 'Seattle', 'San Francisco', 'Denver', 'Portland'],
    'Central': ['Kansas City', 'St. Louis', 'Indianapolis', 'Columbus', 'Oklahoma City'],
}

segments   = ['Consumer', 'Corporate', 'Home Office']
ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

first_names = ['James','Mary','John','Patricia','Robert','Jennifer','Michael','Linda',
               'William','Barbara','David','Elizabeth','Richard','Susan','Joseph','Jessica',
               'Thomas','Sarah','Charles','Karen','Daniel','Lisa','Matthew','Nancy',
               'Anthony','Betty','Mark','Margaret','Donald','Sandra','Steven','Ashley']
last_names  = ['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis',
               'Rodriguez','Martinez','Hernandez','Lopez','Gonzalez','Wilson','Anderson',
               'Thomas','Taylor','Moore','Jackson','Martin','Lee','Thompson','White','Harris']

# Build 2 000 customers with fixed attributes
n_customers = 2000
customer_ids = [f'CUST-{i:05d}' for i in range(1, n_customers + 1)]

customers = {}
for cid in customer_ids:
    region = random.choice(list(regions_cities.keys()))
    customers[cid] = {
        'name':    f'{random.choice(first_names)} {random.choice(last_names)}',
        'region':  region,
        'city':    random.choice(regions_cities[region]),
        'segment': random.choice(segments),
    }

# Month weights: Q4 seasonality
month_weights = [.06,.06,.07,.07,.08,.08,.07,.08,.09,.10,.12,.12]

records = []
for i in range(N):
    month  = np.random.choice(range(1, 13), p=month_weights)
    year   = np.random.choice([2022, 2023, 2024], p=[.30, .35, .35])
    max_day = 28 if month == 2 else (30 if month in [4,6,9,11] else 31)
    day    = random.randint(1, max_day)
    # Cap 2024 at Dec
    if year == 2024 and month == 12 and day > 31:
        day = 31
    order_date = datetime(year, month, day)

    cid     = random.choice(customer_ids)
    cinfo   = customers[cid]
    cat     = random.choice(list(catalog.keys()))
    sub_cat = random.choice(list(catalog[cat].keys()))
    lo, hi  = catalog[cat][sub_cat]

    unit_price = round(random.uniform(lo, hi), 2)
    quantity   = random.randint(1, 10)

    if cat == 'Furniture' or quantity > 5:
        discount = random.choice([0, 0, .10, .10, .20, .20, .30, .40])
    else:
        discount = random.choice([0, 0, 0, 0, .10, .10, .20, .30])

    sales = round(unit_price * quantity * (1 - discount), 2)

    if cat == 'Technology':
        base_margin = random.uniform(.08, .40)
    elif cat == 'Furniture':
        base_margin = random.uniform(-.05, .30)
    else:
        base_margin = random.uniform(.15, .50)

    margin = base_margin - discount * 0.5
    profit = round(sales * margin, 2)

    shipping_cost = round(random.uniform(3, 60), 2)

    records.append({
        'order_id':      f'ORD-{20000 + i}',
        'order_date':    order_date.strftime('%Y-%m-%d'),
        'customer_id':   cid,
        'customer_name': cinfo['name'],
        'segment':       cinfo['segment'],
        'region':        cinfo['region'],
        'city':          cinfo['city'],
        'category':      cat,
        'sub_category':  sub_cat,
        'product_name':  f'{sub_cat} - Model {chr(random.randint(65,90))}{random.randint(100,999)}',
        'quantity':      quantity,
        'unit_price':    unit_price,
        'discount':      discount,
        'sales':         sales,
        'profit':        profit,
        'shipping_cost': shipping_cost,
        'ship_mode':     random.choice(ship_modes),
    })

df = pd.DataFrame(records)

# Introduce ~0.5% missing values (realistic noise)
missing_idx = np.random.choice(df.index, size=int(N * 0.005), replace=False)
df.loc[missing_idx[:len(missing_idx)//2], 'discount']       = np.nan
df.loc[missing_idx[len(missing_idx)//2:], 'shipping_cost']  = np.nan

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), 'sales_data.csv')
df.to_csv(out, index=False)

print(f"Generated {len(df):,} rows x {len(df.columns)} columns → {out}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\nSample stats:\n{df[['sales','profit','quantity','unit_price','discount']].describe().round(2)}")
