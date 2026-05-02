# main.py
# Demonstrating a Failed ML Approach on the 20 Newsgroups Dataset
# ---------------------------------------------------------------
# FAILED METHOD CHOICE: Option A — K-Nearest Neighbors (KNN)
#
# KNN is a reasonable general-purpose classifier, but it is a
# poor fit for high-dimensional sparse text data. This script
# trains both Naive Bayes (baseline) and KNN on the same TF-IDF
# features and shows why KNN underperforms — then explains why.
# ---------------------------------------------------------------

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# TASK 1: DATASET
# ---------------------------------------------------------------

# Use a manageable subset of 5 categories
CATEGORIES = [
    'rec.sport.hockey',
    'sci.space',
    'talk.politics.guns',
    'comp.graphics',
    'rec.autos'
]

print("=" * 60)
print("Loading 20 Newsgroups dataset...")
print(f"Categories: {CATEGORIES}")
print("=" * 60)

# Load all documents (train + test combined) so we control the split
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=CATEGORIES,
    shuffle=True,
    random_state=42,
    remove=('headers', 'footers', 'quotes')  # strip metadata to avoid leakage
)

texts  = newsgroups.data    # list of raw text strings
labels = newsgroups.target  # integer class labels

print(f"\nTotal documents loaded : {len(texts)}")
print(f"Number of categories  : {len(CATEGORIES)}")
print(f"Class distribution    : {dict(zip(*np.unique(labels, return_counts=True)))}\n")

# ---------------------------------------------------------------
# TASK 2: PREPROCESSING — TF-IDF with stopword removal
# ---------------------------------------------------------------

# TF-IDF converts text into a numerical matrix.
# Each document becomes a row; each unique word is a column.
# With 10,000 features, this produces a very large, sparse matrix —
# which is exactly what makes KNN struggle later.
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(texts)
y = labels

sparsity = 100 * (1 - X.nnz / (X.shape[0] * X.shape[1]))

print(f"TF-IDF matrix shape: {X.shape}  (documents x features)")
print(f"Matrix sparsity    : {sparsity:.1f}% zeros\n")

# Single train/test split — same split used for BOTH models
# so the comparison is perfectly fair
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size : {X_train.shape[0]} documents")
print(f"Test set size     : {X_test.shape[0]} documents\n")

# ---------------------------------------------------------------
# TASK 3: BASELINE MODEL — Multinomial Naive Bayes
# ---------------------------------------------------------------

print("=" * 60)
print("BASELINE: Multinomial Naive Bayes")
print("=" * 60)

# Naive Bayes is well-suited to text classification because it
# treats each feature (word) independently and works naturally
# with high-dimensional, sparse TF-IDF data.
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print(f"Baseline (Naive Bayes) Accuracy: {nb_accuracy:.4f}\n")

# ---------------------------------------------------------------
# TASK 4: FAILED METHOD — K-Nearest Neighbors (KNN)
# ---------------------------------------------------------------

print("=" * 60)
print("FAILED METHOD: K-Nearest Neighbors (KNN, k=5)")
print("=" * 60)
print("Note: KNN on high-dimensional TF-IDF data is slow.")
print("This may take a minute...\n")

# KNN classifies a document by finding its k nearest neighbors
# in feature space and taking a majority vote.
# With 10,000 dimensions, "distance" becomes nearly meaningless —
# a known failure mode called the "curse of dimensionality."
knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print(f"Failed Method (KNN) Accuracy: {knn_accuracy:.4f}\n")

# ---------------------------------------------------------------
# TASK 5: COMPARISON
# ---------------------------------------------------------------

difference = nb_accuracy - knn_accuracy

print("=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)
print(f"  Baseline  — Naive Bayes : {nb_accuracy:.4f}")
print(f"  Failed    — KNN (k=5)   : {knn_accuracy:.4f}")
print(f"  Difference (NB - KNN)   : {difference:+.4f}")

if difference > 0:
    pct = (difference / nb_accuracy) * 100
    print(f"  Naive Bayes is {pct:.1f}% more accurate than KNN.")
else:
    print("  KNN unexpectedly matched or beat Naive Bayes.")

print("=" * 60)

# ---------------------------------------------------------------
# TASK 6: FAILURE ANALYSIS
# ---------------------------------------------------------------

failure_analysis = f"""
FAILURE ANALYSIS
----------------

1. WHAT FAILED
   K-Nearest Neighbors (KNN, k=5) achieved {knn_accuracy:.4f} accuracy,
   compared to {nb_accuracy:.4f} for Multinomial Naive Bayes — a gap of
   {difference:.4f} accuracy points. This is a meaningful performance drop
   for a task where the categories are reasonably well-separated.

2. WHY IT FAILED

   a) The Curse of Dimensionality
      KNN works by finding the k training documents "closest" to each
      test document using a distance metric. Our TF-IDF matrix has
      10,000 features (dimensions). In very high-dimensional spaces,
      distances between all points tend to converge. There is almost
      no meaningful difference between a "close" neighbor and a "far"
      one. When neighbors are indistinguishable, majority voting breaks
      down and predictions become unreliable.

   b) Extreme Sparsity
      TF-IDF vectors are extremely sparse. Most entries are zero
      because most words don't appear in any given document. This
      dataset's matrix is {sparsity:.1f}% zeros. Cosine similarity
      (used here to partially mitigate sparsity) helps, but two
      documents can share almost no vocabulary and still discuss the
      same topic — making distance-based reasoning fundamentally weak.

   c) No Model of Language
      Naive Bayes estimates the probability that each word appears
      given a class label. Words like "puck" are strong signals for
      hockey, "orbit" for space. KNN has no such model; it relies
      purely on geometric proximity, which is poorly suited to text.

   d) Slow Prediction
      KNN has no real training phase. It memorizes the training data.
      At prediction time it computes distances to every training
      document for every test document. This is noticeably slower
      than Naive Bayes, which predicts in near-instant time.

3. LESSONS LEARNED

   - Matching the algorithm to the data structure matters as much as
     hyperparameter tuning. A poorly matched model will underperform
     regardless of how carefully it is configured.

   - Text data is high-dimensional and sparse by nature. Distance-based
     algorithms (KNN, k-means, RBF-SVM) tend to struggle unless the
     feature space is reduced first (e.g., via SVD / Latent Semantic
     Analysis).

   - Naive Bayes is a strong text classification baseline because its
     independence assumption aligns well with how TF-IDF features are
     constructed. It is fast, interpretable, and competitive even
     against more complex models on this type of data.

   - Always establish a simple, well-matched baseline before reaching
     for more sophisticated methods. Understanding *why* a model works
     (or fails) is more valuable than blindly trying alternatives.
"""

print(failure_analysis)