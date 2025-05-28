import numpy as np
from sklearn.metrics import precision_score

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