import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    path = Path(args.data)

    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit("CSV must contain text and label columns.")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    if df["label"].nunique() < 2:
        raise SystemExit("Dataset must contain at least two classes.")

    if len(df) < 10:
        raise SystemExit("Please use at least 10 labeled examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Evaluation results:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))


if __name__ == "__main__":
    main()
