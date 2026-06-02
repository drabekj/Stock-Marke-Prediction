"""Standalone TF2 training script for the S&P 500 direction classifier.

Mirrors the logic from `Globe+Markets.ipynb` but written for modern Keras so
it runs on TensorFlow 2.x without the deprecated `tf.contrib` and
`tf.placeholder` APIs.

Usage:
    python train.py
    python train.py --data data/ --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TICKERS = ["^AORD", "^N225", "^HSI", "^GDAXI", "^NYA", "^DJI", "^GSPC"]
TARGET = "^GSPC"  # predict S&P 500 direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="Directory with index CSVs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=3, help="Days of history to use")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_closing_prices(data_dir: Path) -> pd.DataFrame:
    frames = {}
    for ticker in TICKERS:
        path = data_dir / f"{ticker}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        frames[ticker] = df[col].astype(float)
    closing = pd.DataFrame(frames).sort_index()
    closing = closing.ffill().dropna()
    return closing


def make_features(closing: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    returns = closing.pct_change().dropna()
    feature_tickers = [t for t in TICKERS if t != TARGET]

    feature_frames = [returns[feature_tickers].shift(i) for i in range(lookback)]
    features = pd.concat(feature_frames, axis=1).dropna()
    target = (returns[TARGET].loc[features.index] > 0).astype(int)

    return features.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.int32)


def build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    closing = load_closing_prices(Path(args.data))
    X, y = make_features(closing, args.lookback)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    preds = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nTest accuracy: {acc:.4f}  (majority-class baseline: {baseline:.4f})")


if __name__ == "__main__":
    main()
