# Failed ML Approach: KNN vs Naive Bayes on Text Data

A beginner-friendly Python script that demonstrates why K-Nearest Neighbors (KNN) fails on high-dimensional text data, using the 20 Newsgroups dataset as a case study.

---

## What It Shows

| Model             | Expected Performance | Reason                                      |
|-------------------|----------------------|---------------------------------------------|
| Multinomial NB    | Strong baseline      | Designed for sparse, high-dimensional text  |
| KNN (k=5, cosine) | Poor                 | Curse of dimensionality on 10,000 features  |

Both models are trained and evaluated on the **same train/test split** so the comparison is fair.

---

## Requirements

- Python 3.7+
- `scikit-learn`
- `numpy`

Install dependencies with:

```bash
pip install scikit-learn numpy
```

---

## Usage

```bash
python main.py
```

The dataset (~14 MB) is downloaded automatically on the first run and cached locally by sklearn.

**Estimated runtime:** 2–4 minutes. KNN is intentionally slow on high-dimensional data — that slowness is part of the lesson.

---

## Output

All output is printed to the console:

- Dataset and feature matrix stats (shape, sparsity)
- Baseline (Naive Bayes) accuracy
- Failed method (KNN) accuracy
- Side-by-side comparison with percentage difference
- Multi-part failure analysis covering what failed, why it failed, and lessons learned