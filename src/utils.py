import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def calculate_custom_score(y_true, y_pred_proba, n_features):
    """
    Calculate the custom score for the energy usage prediction task.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities (must be raw probabilities, not binary predictions)
    n_features : int
        Number of features used in the model
        
    Returns:
    --------
    float
        The final score (reward - cost)
    """

    # Constants
    target_predictions = 1000  # Number of households to select
    reward_per_hit = 10        # EUR per correct prediction
    cost_per_feature = 200     # EUR per feature
    
    # Get indices of top predictions
    top_indices = np.argsort(y_pred_proba)[-target_predictions:]
    
    # Create binary prediction array
    binary_pred = np.zeros_like(y_pred_proba, dtype=int)
    binary_pred[top_indices] = 1
    
    # Calculate number of correct predictions
    correct_predictions = np.sum((binary_pred == 1) & (y_true == 1))
    
    # Calculate reward and cost
    reward = correct_predictions * reward_per_hit
    cost = n_features * cost_per_feature
    
    # Calculate total score
    total_score = reward - cost
    
    return total_score

def evaluate_model(model, X_train, y_train, X_val, y_val, n_features=None):
    """
    Evaluates a trained model on both training and validation sets and returns a DataFrame with metrics.
    
    Parameters:
    -----------
    model : trained sklearn model
        The fitted model to evaluate
    X_train : pandas DataFrame or numpy array
        Training features
    y_train : pandas Series or numpy array
        Training target values
    X_val : pandas DataFrame or numpy array
        Validation features
    y_val : pandas Series or numpy array
        Validation target values
    n_features : int, optional
        Number of features used in the model, for calculating expected gain. 
        If None, uses all features in X_train.
        
    Returns:
    --------
    pandas DataFrame
        Table with evaluation metrics for both training and validation sets
    """
    
    # If n_features is not provided, use the number of columns in X_train
    if n_features is None:
        n_features = X_train.shape[1]
    
    # Predict on training and validation sets
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Create DataFrame with metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Expected Gain'],
        'Training': [
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            recall_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred),
            roc_auc_score(y_train, y_train_pred_proba),
            calculate_custom_score(y_train, y_train_pred_proba, n_features)
        ],
        'Validation': [
            accuracy_score(y_val, y_val_pred),
            precision_score(y_val, y_val_pred),
            recall_score(y_val, y_val_pred),
            f1_score(y_val, y_val_pred),
            roc_auc_score(y_val, y_val_pred_proba),
            calculate_custom_score(y_val, y_val_pred_proba, n_features)
        ]
    })

    # Format the numbers to 4 decimal places
    metrics_df['Training'] = metrics_df['Training'].map('{:.4f}'.format)
    metrics_df['Validation'] = metrics_df['Validation'].map('{:.4f}'.format)
    
    return metrics_df