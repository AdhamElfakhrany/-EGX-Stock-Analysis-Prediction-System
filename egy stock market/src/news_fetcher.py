"""NewsAPI integration helpers for EGX bank news."""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

from typing import Any

BANK_CONFIG: Dict[str, Dict[str, Any]] = {
    "CIB": {
        "full_name": "Commercial International Bank Egypt",
        "short": ["CIB"],
        "aliases": ["CIB Egypt", "COMI"],
    },
    "FAISAL": {
        "full_name": "Faisal Islamic Bank Egypt",
        "short": ["Faisal"],
        "aliases": ["Faisal Bank Egypt"],
    },
    "HDB": {
        "full_name": "Housing and Development Bank Egypt",
        "short": ["HDB"],
        "aliases": ["Housing Bank Egypt"],
    },
    "ADIB": {
        "full_name": "Abu Dhabi Islamic Bank Egypt",
        "short": ["ADIB"],
        "aliases": ["ADIB Egypt"],
    }
}

FINANCIAL_KEYWORDS: List[str] = ["bank", "stock", "EGX", "finance", "earnings"]


def score_article(article: Dict[str, str], bank_key: str) -> int:
    title = article.get("title") or ""
    description = article.get("description") or ""
    text = (title + " " + description).lower()
    score = 0
    
    config = BANK_CONFIG.get(bank_key)
    if not config:
        return 0

    full_name = str(config.get("full_name", "")).lower()
    aliases = [str(a).lower() for a in config.get("aliases", [])]
    short_names = [str(s).lower() for s in config.get("short", [])]

    if full_name and full_name in text:
        score += 4

    if any(alias in text for alias in aliases):
        score += 3

    if any(short in text for short in short_names):
        score += 2

    if "egypt" in text:
        score += 1

    if any(kw.lower() in text for kw in FINANCIAL_KEYWORDS):
        score += 1

    return score


def _fetch_news_by_query(query: str) -> List[Dict[str, str]]:
    """
    Fetch clean, structured news articles for a query via NewsAPI.

    Env vars:
        NEWSAPI_KEY: API key for https://newsapi.org/
    """
    print("Current working directory:", os.getcwd())
    print("NEWS_API_KEY loaded:", "yes" if API_KEY else "no")
    logger.info("News API key loaded: %s", "yes" if API_KEY else "no")
    if not API_KEY:
        logger.warning("Missing NEWS_API_KEY in environment/.env. Returning empty result.")
        return []

    if not query or not query.strip():
        logger.warning("Empty news query provided. Returning empty result.")
        return []

    params = {
        "q": query.strip(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": API_KEY,
    }
    request_url = requests.Request("GET", NEWSAPI_URL, params=params).prepare().url
    safe_request_url = request_url.replace(API_KEY, "***MASKED***") if API_KEY else request_url
    logger.info("NewsAPI request URL: %s", safe_request_url)

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=20)
        if response.status_code != 200:
            logger.error(
                "NewsAPI request failed for query '%s' with status %s: %s",
                query,
                response.status_code,
                response.text[:300],
            )
            return []
        payload = response.json()
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed for query '%s': %s", query, exc)
        return []
    except ValueError as exc:
        logger.error("Invalid JSON response from NewsAPI for query '%s': %s", query, exc)
        return []

    if payload.get("status") != "ok":
        logger.error(
            "NewsAPI returned non-ok status for query '%s': %s",
            query,
            payload.get("message", "unknown error"),
        )
        return []

    articles = payload.get("articles", [])
    if not articles:
        logger.info("No articles found for query '%s'.", query)
        return []
    logger.info("NewsAPI returned %d articles for query '%s'.", len(articles), query)

    cleaned: List[Dict[str, str]] = []
    for article in articles:
        cleaned.append(
            {
                "title": article.get("title") or "",
                "description": article.get("description") or "",
                "publishedAt": article.get("publishedAt") or "",
                "source": (article.get("source") or {}).get("name") or "",
            }
        )

    return cleaned


def fetch_news(stock_name: str) -> List[Dict[str, str]]:
    """
    Fetch news for one configured bank by stock name.
    """
    key = (stock_name or "").upper().strip()
    config = BANK_CONFIG.get(key)
    
    if not config:
        # Backward compatibility for raw queries
        query = stock_name.strip()
        if not query:
            return []
        articles = _fetch_news_by_query(query)
        return articles[:10]

    full_name = str(config.get("full_name", ""))
    aliases = config.get("aliases", [])
    
    bank_query_parts = [f'"{full_name}"']
    for alias in aliases:
        bank_query_parts.append(f'"{alias}"')
        
    bank_query = " OR ".join(bank_query_parts)
    sector_query = "Egypt banking sector OR Egypt banks OR EGX banks"
    
    bank_articles = _fetch_news_by_query(bank_query)
    for a in bank_articles:
        a["_is_bank"] = True
        
    sector_articles = _fetch_news_by_query(sector_query)
    for a in sector_articles:
        a.setdefault("_is_bank", False)
        
    all_articles = bank_articles + sector_articles
    
    seen_titles = set()
    deduped_articles = []
    for article in all_articles:
        title_lower = (article.get("title") or "").lower().strip()
        if not title_lower or title_lower in seen_titles:
            continue
        seen_titles.add(title_lower)
        deduped_articles.append(article)
        
    total_fetched = len(deduped_articles)
    
    for a in deduped_articles:
        a["score"] = score_article(a, key)
        
    bank_specific = [a for a in deduped_articles if a.get("_is_bank")]
    sector_only = [a for a in deduped_articles if not a.get("_is_bank")]
    
    bank_specific.sort(key=lambda x: x.get("score", 0), reverse=True)
    sector_only.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    selected = []
    selected_bank = 0
    selected_sector = 0
    
    for a in bank_specific:
        if len(selected) < 10:
            selected.append(a)
            selected_bank += 1
            
    for a in sector_only:
        if len(selected) < 10:
            selected.append(a)
            selected_sector += 1
            
    # Clean up internal key
    for a in selected:
        a.pop("_is_bank", None)
    
    print("\n[NEWS DISTRIBUTION]")
    print(f"Stock: {key}")
    print(f"Bank-specific: {selected_bank}")
    print(f"Sector: {selected_sector}\n")

    logger.info(
        "Filtered relevant articles for stock '%s': %d/%d",
        key,
        len(selected),
        total_fetched,
    )

    return selected


def fetch_news_for_bank(bank_name: str) -> List[Dict[str, str]]:
    """Backward-compatible alias for bank-based calls."""
    return fetch_news(bank_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print(f"\n{'=' * 60}\nCIB NEWS TEST\n{'=' * 60}")
    cib_items = fetch_news("CIB")
    print(f"Articles fetched: {len(cib_items)}")
    for i, item in enumerate(cib_items[:5], start=1):
        print(f"{i}. {item['title']} | {item['source']} | {item['publishedAt']}")
