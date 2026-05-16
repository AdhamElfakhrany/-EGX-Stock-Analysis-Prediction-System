"""Sentiment analysis helpers for news articles."""

from __future__ import annotations

from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment(news_list: List[Dict[str, str]]) -> Dict[str, float | int]:
    """
    Analyze sentiment from a list of news items.

    Each news item is expected to contain:
    - title
    - description

    Returns:
        {
            "avg_sentiment": float,
            "positive": int,
            "negative": int,
            "count": int
        }
    """
    if not news_list:
        return {
            "avg_sentiment": 0.0,
            "positive": 0,
            "negative": 0,
            "count": 0,
        }

    analyzer = SentimentIntensityAnalyzer()

    scores: List[float] = []
    positive_count = 0
    negative_count = 0

    for article in news_list:
        if not isinstance(article, dict):
            continue

        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        text = f"{title} {description}".strip()

        if not text:
            continue

        compound = analyzer.polarity_scores(text)["compound"]
        scores.append(compound)

        if compound > 0:
            positive_count += 1
        elif compound < 0:
            negative_count += 1

    if not scores:
        return {
            "avg_sentiment": 0.0,
            "positive": 0,
            "negative": 0,
            "count": 0,
        }

    avg_sentiment = sum(scores) / len(scores)
    return {
        "avg_sentiment": float(avg_sentiment),
        "positive": int(positive_count),
        "negative": int(negative_count),
        "count": int(len(scores)),
    }
