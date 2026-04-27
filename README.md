# Machine-Learning

![AU5v](https://user-images.githubusercontent.com/72096831/200458933-18bfb380-c3bf-436f-bf1e-f109dc2bdd3c.gif)

Notes and implementations from my ML training. Covers core algorithms from basics to time series, with both from-scratch implementations and scikit-learn usage.

## Contents

### Week 1 - Introduction to Machine Learning
- What is Machine Learning
- Types: Supervised, Unsupervised, Reinforcement Learning
- Train/Test Split
- Bias-Variance Tradeoff
- Overfitting and Underfitting
- Model Evaluation basics
- Classification vs Regression vs Clustering examples
- Data Preprocessing (missing values, encoding, scaling, pipelines)

### Week 2 - Python for Data Science
- **Numpy & Pandas**: Array operations, data manipulation, file handling
- **Matplotlib**: Line plots, bar charts, scatter plots, histograms, box plots
- **Seaborn**: Statistical visualizations, pair plots, heatmaps

### Week 3 - Linear Regression & Feature Engineering
- Linear Regression from scratch (gradient descent implementation)
- Linear Regression with scikit-learn
- Gradient Descent variants (Batch, Stochastic, Mini-batch)
- Feature Scaling and Feature Selection
- Encoding Techniques (Label, One-Hot)
- Ridge and Lasso Regression (L1/L2 Regularization)
- OLS assumptions and QQ plots

### Week 4 - Classification
- Logistic Regression (from scratch + sklearn)
- Decision Trees (from scratch + sklearn)
- Random Forest
- Classification metrics (Precision, Recall, F1, ROC-AUC)
- Post Pruning techniques

### Week 5 - Ensemble Methods & Unsupervised Learning
- Gradient Boosting, AdaBoost, XGBoost, LightGBM
- K-Means Clustering (from scratch + sklearn)
- Hierarchical Clustering
- PCA (Dimensionality Reduction)
- Cross Validation
- Missing Value Imputation

### Week 6 - Project
- End-to-end ML project implementation

### Week 7 - Advanced Topics
- K-Nearest Neighbors (KNN)
- Handling Imbalanced Datasets (SMOTE, undersampling)
- Anomaly Detection (Isolation Forest, PyCaret)

### Week 8 - Time Series
- Stationarity and ADF test
- ACF & PACF plots
- Time Series Decomposition
- Smoothing techniques
- AR, ARMA, ARIMA, SARIMA models
- Rolling Forecast

## Setup

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm statsmodels jupyter
```

## Datasets

Week 1 notebooks use sklearn built-in datasets (no download needed):
- `load_iris()` - Classification
- `load_breast_cancer()` - Binary Classification  
- `fetch_california_housing()` - Regression
- `load_digits()` - Multi-class Classification

Other weeks use local CSV files. You can substitute with any similar dataset.

## Notes

- Most algorithms have two implementations: from scratch (to understand the math) and using sklearn (for practical use)
- Notebooks are meant to be run in Jupyter
