# Credit Card Fraud Detection

## Problem Statement
This project evaluates various machine learning models for detecting fraudulent credit card transactions, comparing their performance on a standardized test set.

## Dataset Description
- **Test samples**: 100
- **Number of features**: 12
- **Class distribution**: 5% fraudulent transactions


## Models Used

### Performance Comparison

| ML Model Name         | Accuracy | AUC-ROC | Precision | Recall | F1 Score | MCC    |
|-----------------------|----------|---------|-----------|--------|----------|--------|
| XGBoost               | 1.0000   | 1.0000  | 1.0000    | 1.0000 | 1.0000   | [MCC]  |
| Random Forest         | 1.0000   | 1.0000  | 1.0000    | 1.0000 | 1.0000   | [MCC]  |
| Decision Tree         | 1.0000   | 1.0000  | 1.0000    | 1.0000 | 1.0000   | [MCC]  |
| Naive Bayes           | 1.0000   | 1.0000  | 1.0000    | 1.0000 | 1.0000   | [MCC]  |
| Logistic Regression   | 1.0000   | 1.0000  | 1.0000    | 1.0000 | 1.0000   | [MCC]  |
| kNN                   | 0.9800   | 0.9947  | 1.0000    | 0.6000 | 0.7500   | [MCC]  |

### Model Performance Observations

| ML Model Name         | Observation about model performance |
|-----------------------|-------------------------------------|
| XGBoost               | Achieves perfect performance, suggesting either excellent feature engineering or potential data leakage. The model demonstrates perfect classification on the test set. |
| Random Forest         | Matches XGBoost's performance, indicating that ensemble methods are highly effective for this dataset. |
| Decision Tree         | Perfect performance suggests the data might be perfectly separable using a simple tree structure, though this is unusual for real-world credit card data. |
| Naive Bayes           | Perfect performance is unexpected given its simplicity, suggesting strong feature independence or engineered features. |
| Logistic Regression   | Perfect linear separation indicates the classes are likely linearly separable in the feature space. |
| kNN                   | Shows more realistic performance with 98% accuracy. The 100% precision but 60% recall indicates high confidence in positive predictions but misses some fraud cases. |

## Critical Analysis

### Data Quality Concerns
1. **Unrealistic Performance**:
   - 5/6 models achieve 100% accuracy
   - Suggests potential data leakage or overfitting
   - kNN's lower recall (0.6) is the only indicator of a more realistic scenario

2. **Potential Issues**:
   - Test set might not be properly separated from training
   - Features might contain target information
   - Dataset might be synthetic or heavily preprocessed

### Recommendations

1. **Immediate Actions**:
   - Verify data splitting procedure
   - Check for data leakage
   - Examine feature importance

2. **Model-Specific Improvements**:
   - For kNN: Tune k and distance metric
   - For all models: Implement cross-validation
   - Add regularization to prevent overfitting

3. **Next Steps**:
   - Calculate MCC values
   - Generate learning curves
   - Perform error analysis

## Setup and Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt