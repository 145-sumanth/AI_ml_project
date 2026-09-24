"""Create the SQLite database that stores the cleaned book dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "cleaned_books.csv"
DB_PATH = Path(__file__).resolve().parent / "books.db"


def create_database() -> None:
    """Create categories and books tables and load the cleaned data."""
    df = pd.read_csv(CLEAN_PATH)
    if df.empty:
        raise ValueError("The cleaned dataset is empty; run clean.py first.")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("DROP TABLE IF EXISTS books")
        connection.execute("DROP TABLE IF EXISTS categories")

        connection.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            )
            """
        )

        unique_categories = sorted(df["category"].dropna().unique().tolist())
        for category_name in unique_categories:
            connection.execute(
                "INSERT INTO categories (category_name) VALUES (?)",
                (category_name,),
            )

        category_lookup = {
            row[1]: row[0]
            for row in connection.execute(
                "SELECT category_id, category_name FROM categories"
            ).fetchall()
        }

        connection.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            )
            """
        )

        for row in df.itertuples(index=False):
            category_id = category_lookup[row.category]
            connection.execute(
                """
                INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row.title, row.price_gbp, row.price_inr, int(row.rating), int(bool(row.in_stock)), category_id),
            )

        connection.commit()

    print(f"SQLite database created at {DB_PATH}")
    print(f"Stored rows: {len(df)}")


if __name__ == "__main__":
    create_database()
