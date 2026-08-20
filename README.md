# Film-Junky-Union-NLP

Project Overview

The Film Junky Union is developing a system to filter and categorize movie reviews. Your task was to train and evaluate multiple NLP models capable of classifying IMDB reviews as positive or negative.
The project required:
Text preprocessing
Feature engineering
Training multiple ML models
Evaluating performance using F1, ROC‑AUC, APS, and precision‑recall curves
Achieving ≥ 0.85 F1 score on the test set
You exceeded the requirement with multiple models.

📂 Dataset

The dataset contains 47,331 IMDB reviews, each labeled with:
review — raw text
pos — polarity (1 = positive, 0 = negative)
rating — numeric rating
ds_part — train/test split
Additional metadata (title, year, votes, genres)
The dataset is already split into train and test, preventing leakage.

🧹 Data Preprocessing

Text Normalization
All reviews were normalized using:
Lowercasing
Removing digits
Removing punctuation
Removing underscores
Collapsing multiple spaces
Trimming whitespace
This produced a clean review_norm column used across all models.
Lemmatization
Two approaches were explored:
NLTK WordNetLemmatizer
spaCy lemmatization
These were used in different models to compare performance.

📊 Exploratory Data Analysis

Key findings:
Balanced dataset: ~23.7k positive vs ~23.7k negative reviews
Reviews per movie capped at ~30 to avoid bias
Train/test rating distributions nearly identical
Negative reviews cluster around ratings 1–4; positive around 7–10
No reviews with ratings 5 or 6 → users tend to review only when strongly opinionated
EDA confirmed the dataset was well‑structured for binary classification.

🤖 Models Trained

Four models were built and evaluated using a unified evaluation pipeline.
Model 0 — Baseline (Dummy Classifier)
Strategy: predict most frequent class
F1 score: 0.00
Purpose: establish minimum performance threshold
Model 1 — TF‑IDF + Logistic Regression
Preprocessing: TF‑IDF on normalized text
Achieved:
F1 (test): 0.88
ROC‑AUC: 0.95
APS: 0.95
Exceeded project requirement
Strong generalization (train F1 = 0.93)
Model 3 — Lemmatization (NLTK) + TF‑IDF + Logistic Regression
Added lemmatization
Achieved:
F1 (test): 0.88
ROC‑AUC: 0.95
Same performance as Model 1
More linguistically informed preprocessing
Model 4 — Lemmatization + TF‑IDF + LightGBM
Gradient boosting approach
Achieved:
F1 (test): 0.86
ROC‑AUC: 0.94
Slightly lower performance but still above requirement
Tended to classify short reviews as positive → noted in conclusions

📈 Evaluation Metrics

Each model was evaluated using:
F1 Score
Accuracy
ROC‑AUC
Average Precision Score (APS)
Precision‑Recall Curve
Threshold‑sweep F1 analysis
The evaluation pipeline ensured consistent comparison across models.

🧪 Custom Review Testing

Eight custom reviews were passed through all models.
Findings:
Logistic Regression models produced balanced, realistic predictions
LightGBM tended to over‑predict positivity on short texts
spaCy lemmatization improved interpretability but did not outperform TF‑IDF alone

🏆 Final Results

All three main models (1, 3, 4) exceeded the required F1 ≥ 0.85 threshold.
Best Model:

⭐ TF‑IDF + Logistic Regression

F1: 0.88
ROC‑AUC: 0.95
APS: 0.95
Fast, stable, interpretable, and highly effective

🔍 Key Lessons Learned

TF‑IDF remains a strong baseline for text classification
Lemmatization does not always improve performance
Gradient boosting models require careful calibration for short text
Balanced datasets simplify training and reduce the need for resampling
Evaluation across multiple metrics provides deeper insight than F1 alone

🚀 Future Improvements

Add stopword removal to TF‑IDF for cleaner feature space
Use spaCy preprocessing consistently across training and inference
Explore BERT or DistilBERT for contextual embeddings
Calibrate LightGBM probabilities for short‑text stability
