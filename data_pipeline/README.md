# Data Pipeline

This module builds a small, offline book dataset from books.toscrape.com and stores it in SQLite.

## Install and run

```bash
cd /Users/sumanth/pyhton_test
python3 -m pip install -r requirements.txt
cd data_pipeline
python3 scrape.py
python3 clean.py
python3 db.py
python3 queries.py
```

## Cleaning and conversion decisions

- Price is converted to `float` by stripping the `£` symbol and commas before writing `price_gbp`.
- `rating` is normalized to an integer from 1 to 5 using the text value from the CSS class (for example `One` -> `1`).
- `in_stock` is parsed from the availability text such as `In stock (20 available)` and stored as a boolean.
- `price_inr` is computed as `price_gbp * 105.50`, which matches the required conversion rate of 1 GBP = 105.50 INR.
- Numeric parsing failures are handled by median imputation, while rows missing the required title or category are dropped because they cannot be stored faithfully.

## Expected data flow

1. `scrape.py` downloads product cards across three categories and saves the raw output to `data/raw_books.csv`.
2. `clean.py` fixes the types, imputes missing numeric values, and writes `data/cleaned_books.csv`.
3. `db.py` creates `books.db` and loads the cleaned data into the SQLite schema.
4. `queries.py` runs the required SQL checks and saves query outputs under `query_results/`.

## Verification notes

The end-to-end run produced the following concrete outputs from the code:

- `scrape.py` printed: `Scraped 60 valid rows across 3 categories.`
- `clean.py` printed: `Cleaned rows: 60` and `Median price (GBP): 30.39`.
- `db.py` printed: `Stored rows: 60` and created the SQLite database successfully.
- `queries.py` printed: `JOIN validation passed: pandas merge matches the SQL join output.`

These numbers confirm that the pipeline scraped 60 books, cleaned them, loaded them into SQLite, and verified the SQL and pandas join logic against the same data.
