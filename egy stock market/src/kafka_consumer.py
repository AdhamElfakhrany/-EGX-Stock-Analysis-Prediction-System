"""Kafka consumer for processing streamed EGX bank news sentiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.sentiment import analyze_sentiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_consumer(
    topic: str = "news_topic",
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "news_sentiment_group",
) -> None:
    """Consume news messages from Kafka and print sentiment output."""
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
    except KafkaError as exc:
        logger.error("Failed to connect to Kafka broker at %s: %s", bootstrap_servers, exc)
        return

    logger.info(
        "Consumer started. Topic=%s Broker=%s Group=%s",
        topic,
        bootstrap_servers,
        group_id,
    )
    try:
        for message in consumer:
            payload = message.value
            stock = payload.get("stock", "UNKNOWN")
            title = payload.get("title", "")
            description = payload.get("description", "")
            
            if not title and not description:
                continue

            article = {"title": title, "description": description}
            sentiment = analyze_sentiment([article])
            avg_sentiment = sentiment.get("avg_sentiment", 0.0)

            # Write to CSV in append mode
            csv_path = "data/news_stream.csv"
            file_exists = os.path.exists(csv_path)
            try:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, "a", encoding="utf-8") as f:
                    if not file_exists:
                        f.write("stock,sentiment,timestamp\n")
                    timestamp = datetime.now().isoformat()
                    f.write(f"{stock},{avg_sentiment},{timestamp}\n")
            except Exception as e:
                logger.error("Failed to write to csv: %s", e)
                
            logger.info("[%s] Processed sentiment: %.4f", stock, avg_sentiment)
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka news consumer")
    parser.add_argument("--topic", default="news_topic", help="Kafka topic name")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (e.g. localhost:9092)",
    )
    parser.add_argument("--group-id", default="news_sentiment_group", help="Consumer group id")
    args = parser.parse_args()
    run_consumer(topic=args.topic, bootstrap_servers=args.bootstrap_servers, group_id=args.group_id)
