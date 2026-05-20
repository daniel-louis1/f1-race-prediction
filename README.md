# Predicting Formula 1 Race Winners

**CMSC320 — Data Science Final Project | Spring 2026**

Can we predict whether an F1 driver becomes a race winner based solely on their career statistics? This project applies the full data science pipeline — data curation, EDA, hypothesis testing, and machine learning — to historical F1 data spanning 1950–2024.

**Live site:** https://daniel-louis1.github.io/f1-race-prediction/

---

## My Contributions

I (Daniel Louis) was responsible for **sections D and E** — the primary analysis and machine learning modeling:

- Trained and evaluated a **Random Forest Classifier** to predict race winners
- Handled the class imbalance (115 winners vs. 746 non-winners) using `class_weight="balanced"`
- Produced the confusion matrix and feature importance analysis, identifying `avg_points_per_race` and `total_podiums` as the strongest predictors
- Compared RF performance against the SVM models trained by teammates, concluding RF outperformed SVM on winner F1-score (86% vs. 85%)

---

## Project Overview

**Research question:** Can career-level statistics predict whether an F1 driver has ever won a race?

**Dataset:** [Kaggle F1 World Championship 1950–2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — 861 drivers, 26,759 race results

**Pipeline:**
1. **Data Curation** — merged 8 CSV files into a single driver-level dataset with engineered features (win rate, DNF rate, podium rate, age at debut, etc.)
2. **EDA & Hypothesis Testing** — T-test, Z-test, and ANOVA confirming statistically significant differences between winners and non-winners
3. **ML Models** — SVM (4 kernels) and Random Forest Classifier, both achieving 97% accuracy
4. **Visualization** — model agreement scatter plot and uncertainty analysis for edge-case drivers

**Key findings:**
- Race winners score significantly more points per entry (p ≈ 0)
- Drivers debuting after 1980 have a significantly different win rate than earlier-era drivers (p = 0.02)
- Random Forest correctly identified 15 of 18 actual winners in the test set

---

## Tech Stack

- Python: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `scipy`, `statsmodels`
- Data: Kaggle via `kagglehub`
- Models: `SVC` (linear, poly, sigmoid, RBF), `RandomForestClassifier`

## Running the Notebook

Open `Final_Deliverable.ipynb` in Google Colab or Jupyter. The notebook loads data directly from Kaggle via `kagglehub` — no manual CSV downloads needed.
