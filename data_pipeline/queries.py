"""Run SQL queries against the books database and validate a pandas join.

This script prints each query and result, saves the outputs to the query_results
folder, and verifies that the SQL JOIN matches the same join built in pandas.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

DB_PATH = Path(__file__).resolve().parent / "books.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "query_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def save_query_result(name: str, query: str, frame: pd.DataFrame) -> None:
    """Persist the SQL string and its output for inspection."""
    output_path = OUTPUT_DIR / f"{name}.csv"
    frame.to_csv(output_path, index=False)

    text_path = OUTPUT_DIR / f"{name}.txt"
    with text_path.open("w", encoding="utf-8") as file:
        file.write(query)
        file.write("\n\n")
        file.write(frame.to_string(index=False))


def run_query(connection: sqlite3.Connection, name: str, query: str) -> pd.DataFrame:
    """Print the query and return the DataFrame result."""
    print(f"\n--- {name} ---")
    print(query)
    result = pd.read_sql_query(query, connection)
    print(result.to_string(index=False))
    save_query_result(name, query, result)
    return result


def main() -> None:
    """Execute a set of SQL queries and compare SQL and pandas joins."""
    connection = sqlite3.connect(DB_PATH)

    queries = {
        "select_where_order": (
            "SELECT title, price_gbp, rating FROM books WHERE price_gbp >= 20 "
            "ORDER BY price_gbp DESC LIMIT 10;"
        ),
        "distinct_categories": "SELECT DISTINCT category_id FROM books ORDER BY category_id;",
        "join_in_filter": (
            "SELECT b.title, b.price_gbp, c.category_name FROM books b "
            "JOIN categories c ON c.category_id = b.category_id "
            "WHERE c.category_name IN ('Fiction', 'Mystery') "
            "ORDER BY b.price_gbp DESC LIMIT 10;"
        ),
        "between_prices": (
            "SELECT title, price_gbp, rating FROM books WHERE price_gbp BETWEEN 15 AND 30 "
            "ORDER BY price_gbp;"
        ),
        "join_category_summary": (
            "SELECT c.category_name, COUNT(*) AS book_count FROM books b "
            "JOIN categories c ON c.category_id = b.category_id "
            "GROUP BY c.category_name ORDER BY book_count DESC;"
        ),
    }

    results = {
        key: run_query(connection, key, query)
        for key, query in queries.items()
    }

    # Reproduce the SQL join using pandas merge and verify equality after sorting.
    sql_join = pd.read_sql(
        """
        SELECT b.book_id, b.title, b.price_gbp, b.category_id, c.category_name
        FROM books b
        JOIN categories c ON c.category_id = b.category_id
        ORDER BY c.category_name, b.book_id
        """,
        connection,
    )
    books_for_join = pd.read_sql(
        "SELECT book_id, title, price_gbp, category_id FROM books",
        connection,
    )
    category_lookup = pd.read_sql(
        "SELECT category_id, category_name FROM categories",
        connection,
    )
    pandas_join = pd.merge(
        books_for_join,
        category_lookup,
        on="category_id",
        how="inner",
    )
    pandas_join = pandas_join[["book_id", "title", "price_gbp", "category_id", "category_name"]]

    sql_sorted = sql_join.sort_values(["category_name", "book_id"]).reset_index(drop=True)
    pandas_sorted = pandas_join.sort_values(["category_name", "book_id"]).reset_index(drop=True)
    assert_frame_equal(sql_sorted, pandas_sorted)

    print("\nJOIN validation passed: pandas merge matches the SQL join output.")

    # Save the validation output to a file for easy inspection.
    validation_path = OUTPUT_DIR / "join_validation.csv"
    pandas_sorted.to_csv(validation_path, index=False)

    connection.close()


if __name__ == "__main__":
    main()
