# streamlit_app.py
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import plotly.express as px
import joblib

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load the credit card fraud datasets."""
    test_file = "data/credit_card_fraud_test.csv"
    
    try:
        # Load test data
        test_df = pd.read_csv(test_file)
        X_test = test_df.drop('is_fraud', axis=1).values
        y_test = test_df['is_fraud'].map({'Legitimate': 0, 'Fraud': 1}).values
        feature_names = test_df.drop('is_fraud', axis=1).columns.tolist()
        
        return X_test, y_test, feature_names
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

def load_models():
    """Load all trained models from the model directory."""
    models = {}
    model_files = [f for f in os.listdir('model') if f.endswith('.joblib')]
    
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        try:
            models[model_name] = joblib.load(f'model/{model_file}')
        except Exception as e:
            st.warning(f"Could not load model {model_file}: {str(e)}")
    
    return models

def main():
    # Load data and models
    X_test, y_test, feature_names = load_data()
    models = load_models()
    
    if not models:
        st.error("No models found in the model directory. Please train models first.")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)  # Note: In production, you should load the scaler used during training
    
    # UI
    st.title("Credit Card Fraud Detection")
    st.write("""
    This dashboard evaluates pre-trained machine learning models for detecting fraudulent credit card transactions.
    """)
    
    # Dataset info
    st.sidebar.title("Dataset Information")
    st.sidebar.write(f"Test samples: {len(X_test)}")
    st.sidebar.write(f"Number of features: {len(feature_names)}")
    st.sidebar.write(f"Fraud rate: {np.mean(y_test):.2%}")
    
    # Model selection
    st.sidebar.title("Model Evaluation")
    selected_models = st.sidebar.multiselect(
        "Select models to evaluate",
        list(models.keys()),
        default=list(models.keys())[:2] if len(models) > 1 else list(models.keys())
    )
    
    if st.sidebar.button("Evaluate Selected Models"):
        results = []
        
        for name in selected_models:
            model = models[name]
            with st.spinner(f"Evaluating {name}..."):
                start_time = time.time()
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'Model': name.replace('_', ' ').title(),
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                    'AUC-ROC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5,
                    'MCC': matthews_corrcoef(y_test, y_pred),
                    'Inference Time (ms)': round((time.time() - start_time) * 1000, 2)
                }
                results.append(metrics)
        
        if results:
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("Model Performance on Test Set")
            
            # Metrics table
            st.dataframe(
                results_df.style
                .background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC'], 
                                   cmap='YlGnBu')
                .format({
                    'Accuracy': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1 Score': '{:.3f}',
                    'AUC-ROC': '{:.3f}',
                    'MCC': '{:.3f}',
                    'Inference Time (ms)': '{:.2f}'
                })
            )
            
            # Plot metrics
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC']
            for metric in metrics_to_plot:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y=metric,
                    title=f'{metric} Comparison',
                    text=metric,
                    color='Model'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(2)
            for i, name in enumerate(selected_models):
                with cols[i % 2]:
                    model = models[name]
                    y_pred = model.predict(X_test_scaled)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Legitimate', 'Fraud'],
                        y=['Legitimate', 'Fraud'],
                        title=f"{name.replace('_', ' ').title()} - Confusion Matrix",
                        text_auto=True,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No models were evaluated. Please check if any models are selected.")

if __name__ == "__main__":
    main()