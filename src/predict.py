import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def predict_top_households(student_id="STUDENTID", n_households=1000):
    """
    Predict the top n households most likely to exceed the energy usage threshold
    using only feature 2 and a Random Forest model.
    
    Args:
        student_id (str): Student ID to be used in filenames
        n_households (int): Number of households to select (default: 1000)
    
    """
    # Load the data
    print("Loading data...")
    data_path = '../data/scaled/'
    X_train = pd.read_csv(data_path + 'x_train_scaled.txt', delimiter=' ', index_col=0)
    X_val = pd.read_csv(data_path + 'x_val_scaled.txt', delimiter=' ', index_col=0)
    X_test = pd.read_csv(data_path + 'x_test_scaled.txt', delimiter=' ', index_col=0)
    y_train = pd.read_csv(data_path + 'y_train.txt', delimiter=' ', header=None, index_col=0).squeeze()
    y_val = pd.read_csv(data_path + 'y_val.txt', delimiter=' ', header=None, index_col=0).squeeze()
    
    # Combine train and validation sets for final model training
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Select only feature 2
    feature_idx = 2
    feature_name = f'feature_{feature_idx}'
    X_train_selected = X_train_full[[feature_name]]
    X_test_selected = X_test[[feature_name]]
    
    print(f"Training Random Forest model using only {feature_name}...")
    
    # Train Random Forest with parameters from notebook 3
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    rf_model.fit(X_train_selected, y_train_full)
    
    # Get probability estimates for test data
    print("Predicting probabilities for test data...")
    y_test_pred_proba = rf_model.predict_proba(X_test_selected)[:, 1]
    
    # Get indices of top n households based on probability
    top_indices = np.argsort(y_test_pred_proba)[-n_households:][::-1]
    top_probabilities = y_test_pred_proba[top_indices]

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'household_id': top_indices,
        'probability': top_probabilities
    })

    # Create the required output files according to the specified format
    
    # 1. File for customer indexes
    obs_file = f'../{student_id}_obs.txt'
    with open(obs_file, 'w') as f:
        for idx in top_indices:
            f.write(f"{idx}\n")
    print(f"Selected customer indices saved to {obs_file}")
    
    # 2. File for variable indexes
    vars_file = f'../{student_id}_vars.txt'
    with open(vars_file, 'w') as f:
        f.write(f"{feature_idx}\n")
    print(f"Used variable index saved to {vars_file}")
    
    return results_df

if __name__ == "__main__":
    # Replace "STUDENTID" with your actual student ID
    student_id = "STUDENTID"
    
    # Select top 1000 households
    top_households = predict_top_households(student_id, 1000)
    print(f"Selected {len(top_households)} households")
