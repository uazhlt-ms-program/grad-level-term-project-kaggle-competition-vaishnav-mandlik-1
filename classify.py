"""
classify.py - Text Classification for LING 539 Kaggle Competition

Classifies text documents into three categories:
    0: Not a movie/TV show review
    1: Positive movie/TV show review
    2: Negative movie/TV show review

Approach:
    - Text preprocessing (lowercasing, HTML removal, whitespace normalization)
    - TF-IDF vectorization with word-level (1-3) n-grams and
      character-level (2-6) n-grams for robust text representation
    - LinearSVC (Support Vector Machine) classifier with balanced class weights
    - Stratified K-Fold cross-validation for model evaluation

Algorithms used (covered in LING 539):
    - TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction
    - SVM (Support Vector Machine) via LinearSVC for classification

Usage:
    python classify.py              # Run cross-validation + generate submission
    python classify.py --cv-only    # Only evaluate with cross-validation
    python classify.py --submit-only  # Only generate Kaggle submission file

Author: Vaishnav Mandlik
Course: LING 539 - Statistical NLP (Spring 2026)
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
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Clean and normalize raw text for classification.

    Steps:
        1. Convert to lowercase
        2. Strip HTML tags (e.g., <br />)
        3. Remove URLs
        4. Collapse whitespace

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def build_tfidf_features(train_text, test_text):
    """
    Build TF-IDF feature matrices from training and test text.

    Creates two complementary feature sets:
        - Word-level n-grams (1-3): captures content words, bigrams, trigrams
        - Character-level n-grams (2-6): captures morphological patterns,
          spelling variations, and subword information

    Both use sublinear TF (log-scaled term frequency) which is standard
    practice for text classification as it reduces the impact of very
    frequent terms.

    Args:
        train_text (pd.Series): Cleaned training documents.
        test_text (pd.Series): Cleaned test documents.

    Returns:
        tuple: (X_train, X_test) sparse feature matrices.
    """
    # Word-level TF-IDF: unigrams, bigrams, and trigrams
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=200_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )

    # Character-level TF-IDF: 2 to 6 character sequences
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        max_features=200_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    # Fit on training data, transform both sets
    print("  Fitting word-level TF-IDF (1-3 grams)...")
    X_train_word = word_vectorizer.fit_transform(train_text)
    X_test_word = word_vectorizer.transform(test_text)
    print(f"    Word features: {X_train_word.shape[1]:,}")

    print("  Fitting char-level TF-IDF (2-6 grams)...")
    X_train_char = char_vectorizer.fit_transform(train_text)
    X_test_char = char_vectorizer.transform(test_text)
    print(f"    Char features: {X_train_char.shape[1]:,}")

    # Concatenate word and character features
    X_train = hstack([X_train_word, X_train_char])
    X_test = hstack([X_test_word, X_test_char])
    print(f"    Total features: {X_train.shape[1]:,}")

    return X_train, X_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """
    Create a LinearSVC classifier.

    LinearSVC implements a linear Support Vector Machine, which finds an
    optimal separating hyperplane between classes by maximizing the margin.
    It is well-suited for high-dimensional sparse data like TF-IDF vectors.

    Configuration:
        - C=0.8: regularization parameter (lower = more regularization)
        - class_weight='balanced': adjusts weights inversely proportional
          to class frequencies, helping with the class imbalance in our data
          (label 0 is ~46% while labels 1 and 2 are ~27% each)
        - max_iter=5000: sufficient iterations for convergence

    Returns:
        LinearSVC: configured classifier instance.
    """
    return LinearSVC(
        C=0.8,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------
def run_cross_validation(X, y, n_folds=5):
    """
    Evaluate the model using stratified k-fold cross-validation.

    Stratified folds ensure each fold preserves the overall class
    distribution, giving more reliable F1 estimates.

    Args:
        X: Feature matrix (sparse).
        y: Label series.
        n_folds (int): Number of CV folds.

    Returns:
        float: Mean macro F1 score across folds.
    """
    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()

        model = build_model()
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])

        f1 = f1_score(y.iloc[val_idx], preds, average="macro")
        fold_scores.append(f1)
        elapsed = time.time() - fold_start
        print(f"  Fold {fold}/{n_folds}: macro-F1 = {f1:.5f}  ({elapsed:.1f}s)")

    # Print detailed report for last fold
    print(f"\n  Classification Report (Fold {n_folds}):")
    print(classification_report(
        y.iloc[val_idx], preds,
        target_names=["Not review (0)", "Positive (1)", "Negative (2)"],
    ))

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f"  Mean macro-F1: {mean_f1:.5f} (+/- {std_f1:.5f})")
    print(f"{'='*60}\n")
    return mean_f1


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    """Run the full classification pipeline."""
    parser = argparse.ArgumentParser(description="LING 539 Text Classifier")
    parser.add_argument("--cv-only", action="store_true",
                        help="Only run cross-validation (no submission file)")
    parser.add_argument("--submit-only", action="store_true",
                        help="Only generate submission (skip cross-validation)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    total_start = time.time()

    # 1. Load data
    print("\n[Step 1/4] Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train_df["TEXT"] = train_df["TEXT"].fillna("")
    test_df["TEXT"] = test_df["TEXT"].fillna("")
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples:     {len(test_df):,}")
    print(f"  Label distribution: {dict(train_df['LABEL'].value_counts().sort_index())}")

    # 2. Preprocess text
    print("\n[Step 2/4] Preprocessing text...")
    train_df["CLEAN"] = train_df["TEXT"].apply(clean_text)
    test_df["CLEAN"] = test_df["TEXT"].apply(clean_text)
    print("  Done.")

    # 3. Extract TF-IDF features
    print("\n[Step 3/4] Extracting TF-IDF features...")
    X_train, X_test = build_tfidf_features(train_df["CLEAN"], test_df["CLEAN"])

    # 4a. Cross-validation
    if not args.submit_only:
        run_cross_validation(X_train, train_df["LABEL"], n_folds=args.cv_folds)

    if args.cv_only:
        print(f"Total time: {time.time() - total_start:.1f}s")
        return

    # 4b. Train on all data and generate submission
    print("[Step 4/4] Training final model on all data...")
    model = build_model()
    model.fit(X_train, train_df["LABEL"])
    predictions = model.predict(X_test)

    # Save submission CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"ID": test_df["ID"], "LABEL": predictions})
    output_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)

    # Summary
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\n  Prediction distribution: {dict(zip(unique, counts))}")
    print(f"  Submission saved to: {output_path}")
    print(f"  Total time: {time.time() - total_start:.1f}s")
    print("\nDone! Upload output/submission.csv to Kaggle.")


if __name__ == "__main__":
    main()
