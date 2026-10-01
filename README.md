# Machine Learning: Employee Attrition & House Price Prediction

Two end-to-end supervised learning projects built with scikit-learn pipelines, from raw data through cleaning, exploratory analysis, preprocessing, model comparison and evaluation.

| | Employee attrition | House price prediction |
| --- | --- | --- |
| Notebook | [`asn_01_classification_completed.ipynb`](asn_01_classification_completed.ipynb) | [`ai_regression/asn_01_regression_completed.ipynb`](ai_regression/asn_01_regression_completed.ipynb) |
| Task | Binary classification: will an employee leave? | Regression: predict `SalePrice` |
| Data | HR Analytics, 1,480 employees, 38 columns | Ames Housing (Iowa), 1,460 homes, 81 columns |
| Models | DummyClassifier baseline, Logistic Regression, class-weighted Logistic Regression with tuned threshold | Linear Regression, Ridge, Lasso (5-fold CV) |

## Results

### Employee attrition

Only 16% of employees leave, so accuracy is misleading: a majority-class baseline scores 83.8% while catching zero leavers. The model was optimized for **recall** on the leaving class.

| Model (test set, 48 leavers) | Recall | Precision | F1 | Leavers missed |
| --- | --- | --- | --- | --- |
| Logistic Regression | 62.5% | 81.1% | 0.706 | 18 |
| + `class_weight="balanced"` | 79.2% | 40.0% | 0.532 | 10 |
| + threshold tuned to 0.345 on a validation split | **87.5%** | 35.0% | 0.500 | **6** |

Recall rose 40% and missed leavers fell 66.7%, trading precision for fewer missed departures. EDA found employees working overtime leave at 3x the rate of others.

### House prices

| Model | 5-fold CV RMSE |
| --- | --- |
| **Ridge (alpha=10)** | **$29,920** |
| Lasso (alpha=0.05) | $33,387 |
| Linear Regression | $33,749 |

Ridge cut RMSE by 11.3% versus plain linear regression. Final test RMSE: **$31,593** (about 17.5% of the mean sale price).

## Techniques

- `Pipeline` + `ColumnTransformer` for leak-free preprocessing (imputation, scaling, one-hot encoding)
- Stratified train/test split, separate validation split for threshold tuning
- Class weighting and decision-threshold tuning for imbalanced data
- Precision-recall curves, confusion matrices, residual analysis
- Feature engineering: `TotalSF`, `TotalBathrooms`, `AgeAtSale`, `QualityIndex`
- Train-only skewness detection with `log1p` transforms on 22 features
- Custom ARFF parser for string-typed attributes

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Data

- HR Analytics employee attrition dataset (`asn_01_classification_data2.csv`)
- [Ames Housing dataset](https://www.openml.org/search?type=data&status=active&id=42165) (`ai_regression/asn_01_regression_data2.arff`)
