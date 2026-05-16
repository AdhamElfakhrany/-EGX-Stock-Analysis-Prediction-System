"""Kafka producer for streaming EGX bank news."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError

from src.etl import STOCKS
from src.news_fetcher import fetch_news

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_message(stock_name: str, ticker: str, query: str, article: dict) -> dict:
    return {
        "stock": stock_name,
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "publishedAt": article.get("publishedAt", ""),
    }


def run_producer(
    topic: str = "news_topic",
    bootstrap_servers: str = "localhost:9092",
    interval_seconds: int = 30,
    iterations: int = 0,
) -> None:
    """
    Produce EGX bank news messages to Kafka.

    Args:
        topic: Kafka topic name.
        bootstrap_servers: Kafka broker address.
        interval_seconds: Sleep interval between polling cycles.
        iterations: Number of cycles. 0 means run forever.
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8"),
        )
    except KafkaError as exc:
        logger.error("Failed to connect to Kafka broker at %s: %s", bootstrap_servers, exc)
        return

    logger.info("Producer started. Topic=%s Broker=%s", topic, bootstrap_servers)
    cycle = 0
    try:
        while True:
            cycle += 1
            sent_count = 0
            for stock_name, ticker in STOCKS.items():
                try:
                    news_items = fetch_news(stock_name)
                    if not news_items:
                        logger.info("[%s] No news fetched this cycle.", stock_name)
                        continue

                    for article in news_items:
                        message = _build_message(stock_name, ticker, stock_name, article)
                        producer.send(topic=topic, key=stock_name, value=message)
                        sent_count += 1
                    logger.info("[%s] Sent %d messages", stock_name, len(news_items))
                except Exception as exc:
                    logger.error("[%s] Error in producer cycle: %s", stock_name, exc)

            producer.flush()
            logger.info("Cycle %d complete. Total messages sent: %d", cycle, sent_count)

            if iterations > 0 and cycle >= iterations:
                break
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Producer stopped by user.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka news producer")
    parser.add_argument("--topic", default="news_topic", help="Kafka topic name")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (e.g. localhost:9092)",
    )
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of polling cycles (0 = infinite)",
    )
    args = parser.parse_args()
    run_producer(
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        interval_seconds=args.interval,
        iterations=args.iterations,
    )
