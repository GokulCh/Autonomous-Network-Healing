import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_network_data(df):
    """
    Preprocess the UNSW-NB15 network dataset for anomaly detection
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with network traffic features
    
    Returns:
    tuple: Preprocessed features and corresponding labels
    """
    # Remove unnecessary columns
    columns_to_drop = ['id', 'attack_cat', 'label']
    X = df.drop(columns=columns_to_drop)
    y = df['label']
    
    # Handle categorical features
    categorical_cols = ['proto', 'service', 'state']
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns
    )
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X_imputed), 
        columns=X_imputed.columns
    )
    
    # Additional feature engineering for network characteristics
    X_normalized['total_packets'] = X_normalized['spkts'] + X_normalized['dpkts']
    X_normalized['total_bytes'] = X_normalized['sbytes'] + X_normalized['dbytes']
    X_normalized['packet_rate_diff'] = np.abs(X_normalized['spkts'] - X_normalized['dpkts'])
    
    return X_normalized, y

def main():
    # Load the dataset
    train_data = pd.read_csv('unsw_train_data.csv')
    test_data = pd.read_csv('unsw_test_data.csv')
    
    # Preprocess training data
    X_train_processed, y_train = preprocess_network_data(train_data)
    
    # Preprocess testing data
    X_test_processed, y_test = preprocess_network_data(test_data)
    
    # Save processed datasets
    X_train_processed.to_csv('processed_train_features.csv', index=False)
    y_train.to_csv('processed_train_labels.csv', index=False)
    X_test_processed.to_csv('processed_test_features.csv', index=False)
    y_test.to_csv('processed_test_labels.csv', index=False)
    
    print("Data preprocessing completed!")
    print(f"Training features shape: {X_train_processed.shape}")
    print(f"Testing features shape: {X_test_processed.shape}")

if __name__ == '__main__':
    main()