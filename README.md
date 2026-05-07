# LING 539 Text Classification — Kaggle Competition

Text classification system for the LING 539 (Spring 2026) Kaggle competition.  
Classifies documents as **not a review (0)**, **positive review (1)**, or **negative review (2)**.

## Approach

1. **Text Preprocessing** — Lowercasing, HTML tag removal, URL removal, whitespace normalization  
2. **Feature Extraction** — TF-IDF vectorization with:
   - Word-level n-grams (1–3) capturing content words, bigrams, and trigrams
   - Character-level n-grams (2–6) capturing morphological patterns and subword information  
3. **Classification** — LinearSVC (Support Vector Machine) with balanced class weights

**Algorithms covered in LING 539:**  
- TF-IDF (Term Frequency–Inverse Document Frequency)  
- SVM (Support Vector Machine) via `LinearSVC`

## Results

- **Cross-validation macro F1:** ~0.924  
- **Kaggle leaderboard:** Rank 20 (public leaderboard)

## Repository Structure

```
├── classify.py          # Main classification pipeline
├── data/
│   ├── train.csv        # Training data (download from Kaggle)
│   ├── test.csv         # Test data (download from Kaggle)
│   └── sample_submission.csv
├── output/
│   └── submission.csv   # Generated predictions for Kaggle
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition for reproducibility
└── README.md
```

## Setup & Usage

### Option 1: Local (Python 3.8+)

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle and place in data/ directory
# https://www.kaggle.com/competitions/ling-539-competition-2026/data

# Run full pipeline (cross-validation + submission)
python classify.py

# Run cross-validation only
python classify.py --cv-only

# Generate submission only (skip CV)
python classify.py --submit-only
```

### Option 2: Docker

```bash
# Build the container
docker build -t ling539-classifier .

# Run the classifier
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output ling539-classifier python classify.py --submit-only
```

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- scipy

See `requirements.txt` for full list.

## Competition

- **Competition page:** [LING 539 Kaggle Competition](https://www.kaggle.com/competitions/ling-539-competition-2026)
- **Task description:** [Assignment page](https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/)
