"""Scrape and save a raw book dataset from books.toscrape.com.

The site places each product on several pages by category. To keep the pipeline
robust, we intentionally parse only the required fields and skip any row with a
missing or malformed product card instead of crashing the entire run.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "raw_books.csv"


def request_page(url: str) -> requests.Response:
    """Fetch a page with a polite delay and explicit UTF-8 decoding."""
    time.sleep(random.uniform(1.0, 2.5))
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.encoding = "utf-8"
    response.raise_for_status()
    return response


def extract_category(soup: BeautifulSoup) -> str:
    """Read the visible category name from the breadcrumb for a category page."""
    crumbs = [node.get_text(" ", strip=True) for node in soup.select("ul.breadcrumb li")]
    if len(crumbs) >= 2:
        return crumbs[-1]
    return "Uncategorized"


def page_url_for(base_url: str, page_number: int) -> str:
    """Build the correct pagination URL for books.toscrape.com category pages."""
    if page_number == 1:
        return base_url
    if base_url.endswith("/index.html"):
        root = base_url.rsplit("/index.html", 1)[0]
        return f"{root}/page-{page_number}.html"
    return f"{base_url.rstrip('/')}/page-{page_number}.html"


def parse_rating_value(star_rating_tag) -> str:
    """Return the text rating from the p.star-rating CSS class."""
    classes = star_rating_tag.get("class", []) if star_rating_tag else []
    for rating_name in ["One", "Two", "Three", "Four", "Five"]:
        if rating_name in classes:
            return rating_name
    return "Unknown"


def parse_category_page(category_url: str, min_books: int, rows: list[dict]) -> None:
    """Scrape one category across up to five pages until we have enough data."""
    for page_number in range(1, 6):
        if len(rows) >= min_books:
            return

        page_url = page_url_for(category_url, page_number)
        try:
            response = request_page(page_url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("article.product_pod")
        if not cards:
            break

        for card in cards:
            if len(rows) >= min_books:
                break
            try:
                title_tag = card.select_one("h3 a")
                title = title_tag.get("title", "").strip() if title_tag else ""
                price_tag = card.select_one("p.price_color")
                price = price_tag.get_text(strip=True) if price_tag else ""
                rating_tag = card.select_one("p.star-rating")
                star_rating = parse_rating_value(rating_tag)
                availability_tag = card.select_one("p.availability")
                availability = availability_tag.get_text(" ", strip=True) if availability_tag else ""
                category = extract_category(soup)

                if not title or not price or not availability:
                    continue

                rows.append(
                    {
                        "title": title,
                        "price": price,
                        "star_rating": star_rating,
                        "availability": availability,
                        "category": category,
                    }
                )
            except Exception as exc:  # pragma: no cover - intentionally defensive
                print(f"Skipping malformed row in {page_url}: {exc}")
                continue


def fetch_category_urls() -> list[str]:
    """Collect the first three category pages from the home page nav."""
    home_response = request_page(BASE_URL)
    home_soup = BeautifulSoup(home_response.text, "html.parser")
    urls = []
    for link in home_soup.select("ul.nav li a"):
        href = link.get("href", "")
        if "/category/books/" in href:
            urls.append(urljoin(BASE_URL, href))
    unique_urls = list(dict.fromkeys(urls))
    return unique_urls[:3]


def main() -> None:
    """Scrape at least 60 books and save them to a raw CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    categories = fetch_category_urls()
    rows: list[dict] = []

    for category_url in categories:
        parse_category_page(category_url, min_books=60, rows=rows)
        if len(rows) >= 60:
            break

    if len(rows) < 60:
        raise ValueError(f"Only {len(rows)} rows were scraped; at least 60 are required.")

    with RAW_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["title", "price", "star_rating", "availability", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scraped {len(rows)} valid rows across {len(categories)} categories.")
    print(f"Raw dataset saved to {RAW_PATH}")


if __name__ == "__main__":
    main()
