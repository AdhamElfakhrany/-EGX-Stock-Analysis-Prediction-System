"""Machine learning model module for stock price prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SENTIMENT_FEATURES = ["avg_sentiment", "news_count", "positive_news", "negative_news"]


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model and print core classification metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    return metrics


class StockPricePredictor:
    """Stock price direction prediction model using RandomForestClassifier."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use (default: 'random_forest')
            random_state: Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.feature_columns = None
        self.is_fitted = False
        self.last_metrics = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align and clean incoming features to match training schema."""
        if self.feature_columns is None:
            raise ValueError("Model feature columns are not initialized")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        missing_cols = [c for c in self.feature_columns if c not in X.columns]
        extra_cols = [c for c in X.columns if c not in self.feature_columns]
        if missing_cols:
            logging.info("Missing prediction features filled with 0: %s", missing_cols)
        if extra_cols:
            logging.info("Extra prediction features ignored: %s", extra_cols)

        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0)
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        if X_aligned.isna().any().any():
            raise ValueError("NaN values remain in aligned features")
        return X_aligned
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the classification model with 80/20 train/test split.
        
        Args:
            X: Feature matrix
            y: Target vector (1: price up, 0: price down)
            test_size: Test set proportion (default: 0.2)
        """
        logging.info(f"Training model with {X.shape[0]} samples and {X.shape[1]} features")
        if X.empty or y.empty:
            raise ValueError("X and y must be non-empty")
        if y.nunique() < 2:
            raise ValueError("Target must contain at least 2 classes")
        class_distribution = y.value_counts().to_dict()
        logging.info(
            "Samples: %d | Class distribution (DOWN=0, UP=1): %s",
            len(y),
            class_distribution
        )
        
        # Ensure sentiment features are always present in training schema.
        X = X.copy()
        for col in SENTIMENT_FEATURES:
            if col not in X.columns:
                X[col] = 0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        if X.isna().any().any():
            raise ValueError("NaN values detected in training features after cleaning")

        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train/test split (80/20)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback for very small or imbalanced filtered datasets.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
        logging.info(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logging.info("Training RandomForestClassifier...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Print metrics
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Train Accuracy:  {train_accuracy:.4f}")
        print(f"Test Accuracy:   {test_accuracy:.4f}")
        print(f"Precision:       {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
        print(f"Recall:          {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
        print(f"F1-Score:        {f1_score(y_test, y_pred_test):.4f}")
        
        # Independent evaluation function for consistent reporting.
        eval_metrics = evaluate_model(self, X_test, y_test)
        self.last_metrics = {
            "train_accuracy": float(train_accuracy),
            "accuracy": float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall": float(eval_metrics["recall"]),
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "test_samples": int(len(y_test)),
            "train_samples": int(len(y_train)),
        }
        self.print_feature_importance(top_n=15)
        
        print("="*60 + "\n")
        
        logging.info(f"Model training completed. Test Accuracy: {test_accuracy:.4f}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importances from the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before reading feature importance")
        if self.feature_columns is None:
            raise ValueError("Feature columns are not available")

        importances = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        return importances.reset_index(drop=True)

    def print_feature_importance(self, top_n: int = 15) -> None:
        """Print top feature importances for model interpretability."""
        fi = self.get_feature_importance().head(top_n)
        print("\nTop Feature Importances:")
        for _, row in fi.iterrows():
            print(f"{row['feature']:<20} {row['importance']:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for direction (1: up, 0: down).
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            filepath: Path to the model file
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        logging.info(f"Model loaded from {filepath}")


def predict_latest(df: pd.DataFrame, model: StockPricePredictor) -> dict:
    """
    Predict next day stock movement using the latest data.
    
    Args:
        df: DataFrame with engineered features (last row is the latest)
        model: Trained StockPricePredictor model
    
    Returns:
        Dictionary with prediction and probability
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Get the latest row
    latest_row = df.iloc[[-1]]
    
    # Select only feature columns (exclude target if present)
    feature_cols = [col for col in latest_row.columns if col != 'price_up']
    X_latest = latest_row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Make prediction
    prediction = model.predict(X_latest)[0]
    probabilities = model.predict_proba(X_latest)[0]
    
    # Prepare result
    result = {
        'date': latest_row.index[0],
        'latest_price': latest_row['Close'].values[0],
        'prediction': 'UP' if prediction == 1 else 'DOWN',
        'prediction_code': prediction,
        'probability_down': probabilities[0],
        'probability_up': probabilities[1],
        'confidence': max(probabilities)
    }
    
    return result


def retrain_models_per_stock(data_dir: str = "data") -> dict:
    """
    Train one model per processed stock file and print feature importances.

    Expects files like data/CIB_processed.csv.
    """
    import os
    from glob import glob

    pattern = os.path.join(data_dir, "*_processed.csv")
    stock_files = sorted(
        [
            path
            for path in glob(pattern)
            if not path.endswith("processed_stock.csv")
        ]
    )
    if not stock_files:
        raise FileNotFoundError(f"No stock processed files found under {data_dir}")

    trained_models = {}
    for file_path in stock_files:
        stock_name = os.path.basename(file_path).replace("_processed.csv", "")
        logging.info(f"[{stock_name}] Loading training dataset from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if "price_up" not in df.columns:
            logging.warning(f"[{stock_name}] Missing price_up target. Skipping.")
            continue

        X = df.drop(columns=["price_up"], errors="ignore")
        y = df["price_up"]
        model = StockPricePredictor()
        print(f"\n{'='*60}\nTRAINING MODEL FOR {stock_name}\n{'='*60}")
        model.train(X, y)
        trained_models[stock_name] = model

    if not trained_models:
        raise RuntimeError("No models were trained. Check stock data files and target column.")

    return trained_models
