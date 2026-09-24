"""Clean the scraped book dataset and prepare it for SQLite storage.

Numeric parsing problems are handled with median imputation so we do not lose a
large portion of the dataset. Rows with unusable titles or categories are dropped
because those values are required for storage and analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "raw_books.csv"
CLEAN_PATH = DATA_DIR / "cleaned_books.csv"
INDIA_INR_RATE = 105.50


def parse_price_gbp(value) -> float:
    """Convert a string like £23.99 into a float value in GBP."""
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip()
    if not cleaned:
        return np.nan
    try:
        return float(cleaned.replace("£", "").replace(",", ""))
    except ValueError:
        return np.nan


def parse_rating(value) -> int:
    """Map star-rating text to a 1..5 integer value."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return int(value)
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }
    text = str(value).strip()
    if text in rating_map:
        return rating_map[text]
    try:
        return int(float(text))
    except ValueError:
        return np.nan


def parse_in_stock(value) -> bool:
    """Return True when availability text says the book is in stock."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return "in stock" in text


def main() -> None:
    """Load, clean, and save the scraped book data."""
    df = pd.read_csv(RAW_PATH)

    # The title and category columns are required for persistence and analysis, so
    # rows without them are dropped. Numeric parsing issues use median imputation to
    # keep the dataset usable without losing the whole row.
    df = df.dropna(subset=["title", "category"]).copy()
    df["price_gbp"] = df["price"].apply(parse_price_gbp)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_in_stock)

    median_price = df["price_gbp"].median()
    median_rating = df["rating"].median()
    df["price_gbp"] = df["price_gbp"].fillna(median_price)
    df["rating"] = df["rating"].fillna(median_rating).round().astype(int)

    # The INR conversion is fixed by the business requirement and does not require
    # any API or external lookup.
    df["price_inr"] = (df["price_gbp"] * INDIA_INR_RATE).round(2)
    df["in_stock"] = df["in_stock"].fillna(False).astype(bool)
    df["category"] = df["category"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()

    cleaned = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].copy()
    cleaned.to_csv(CLEAN_PATH, index=False)

    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Median price (GBP): {cleaned['price_gbp'].median():.2f}")
    print(f"Saved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
