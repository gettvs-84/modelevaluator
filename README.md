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

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Model Files
All trained models are stored in the `model/` directory as **pickle (.pkl) files**:
- `decision_tree.pkl`
- `random_forest.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `logistic_regression.pkl`
- `xgboost.pkl`

### Data Files
- Training data: `data/credit_card_fraud_dataset.csv`
- Test data: `data/credit_card_fraud_test.csv`

### Running the Application
1. **Streamlit Web App**:
   ```bash
   streamlit run streamlit_app.py
   ```
   
   The app will launch in your browser at `http://localhost:8501`

2. **Features**:
   - Interactive model comparison dashboard
   - Real-time performance metrics
   - Confusion matrix visualization
   - Multi-model evaluation

### Project Structure
```
modelevaluator-main/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Dataset files
│   ├── credit_card_fraud_dataset.csv
│   └── credit_card_fraud_test.csv
└── model/                   # Trained models (.pkl files)
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── logistic_regression.pkl
    └── xgboost.pkl
```

### Dependencies
The application requires the following packages:
- `streamlit==1.32.0` - Web framework
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.1` - Data manipulation
- `scikit-learn==1.4.1.post1` - Machine learning
- `plotly==5.18.0` - Interactive visualizations
- `xgboost==2.0.3` - Gradient boosting models

### Recent Changes
- **Model Format**: Changed from joblib (.joblib) to pickle (.pkl) files for better compatibility
- **Dependencies**: Updated requirements.txt to include only necessary packages
- **Data Path**: Updated to use relative paths for cloud deployment compatibility
- **Performance**: All models retrained and saved in the new format
