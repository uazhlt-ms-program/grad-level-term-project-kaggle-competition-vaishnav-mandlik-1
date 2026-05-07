"""
LING 539 Kaggle Competition - Text Classification
Classify text as: not a review (0), positive review (1), negative review (2)
Uses TF-IDF features with Logistic Regression (Unit 3)
"""

import argparse
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_features(train_text, test_text):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        max_features=200_000, min_df=2, max_df=0.95,
        sublinear_tf=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 6),
        max_features=200_000, min_df=2, max_df=0.95,
        sublinear_tf=True, strip_accents="unicode",
    )

    X_tr_w = word_vec.fit_transform(train_text)
    X_te_w = word_vec.transform(test_text)

    X_tr_c = char_vec.fit_transform(train_text)
    X_te_c = char_vec.transform(test_text)

    return hstack([X_tr_w, X_tr_c]), hstack([X_te_w, X_te_c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train["TEXT"] = train["TEXT"].fillna("")
    test["TEXT"] = test["TEXT"].fillna("")

    print(f"train: {len(train)}, test: {len(test)}")

    train["CLEAN"] = train["TEXT"].apply(clean_text)
    test["CLEAN"] = test["TEXT"].apply(clean_text)

    print("building features...")
    X_train, X_test = get_features(train["CLEAN"], test["CLEAN"])
    print(f"features: {X_train.shape[1]}")

    if not args.submit_only:
        print(f"running {args.cv_folds}-fold cv...")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        scores = []
        for fold, (ti, vi) in enumerate(skf.split(X_train, train["LABEL"]), 1):
            model = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=42)
            model.fit(X_train[ti], train["LABEL"].iloc[ti])
            preds = model.predict(X_train[vi])
            f1 = f1_score(train["LABEL"].iloc[vi], preds, average="macro")
            scores.append(f1)
            print(f"  fold {fold}: {f1:.5f}")
        print(f"mean f1: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")

    if args.cv_only:
        return

    print("training final model...")
    model = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X_train, train["LABEL"])
    predictions = model.predict(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    pd.DataFrame({"ID": test["ID"], "LABEL": predictions}).to_csv(out_path, index=False)

    print(f"saved to {out_path}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

