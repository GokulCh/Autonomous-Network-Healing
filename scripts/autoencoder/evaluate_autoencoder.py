import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Load the saved model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def evaluate_model(model, test_data, threshold_factor=1.5):
    """
    Evaluate the model on test data and calculate anomaly statistics.
    """
    # Reconstruct the input
    reconstructed = model.predict(test_data, verbose=0)

    # Calculate reconstruction errors
    reconstruction_errors = np.mean(np.square(test_data - reconstructed), axis=1)

    # Calculate adaptive threshold
    threshold = reconstruction_errors.mean() + threshold_factor * reconstruction_errors.std()

    # Identify anomalies
    anomalies = reconstruction_errors > threshold

    return reconstruction_errors, anomalies, threshold

def plot_evaluation_results(reconstruction_errors, anomalies, threshold):
    """
    Plot evaluation results including reconstruction error distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50, color='blue', alpha=0.7, label='Reconstruction Errors')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title('Reconstruction Error Distribution on Test Data')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Load processed test dataset
    X_test = pd.read_csv('processed_test_features.csv')
    
    # Load processed train dataset to get the correct columns
    X_train = pd.read_csv('processed_train_features.csv')

    # Ensure X_test has the same columns as X_train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0) 
    X_test = X_test.values # Convert to NumPy array
    

    # Load trained model
    model_path = 'network_anomaly_autoencoder_v2.keras'
    model = load_model(model_path)

    # Evaluate the model
    reconstruction_errors, anomalies, threshold = evaluate_model(model, X_test)

    # Plot evaluation results
    plot_evaluation_results(reconstruction_errors, anomalies, threshold)

    # Print evaluation summary
    total_samples = len(X_test)
    detected_anomalies = np.sum(anomalies)
    anomaly_percentage = (detected_anomalies / total_samples) * 100

    print(f"Total test samples: {total_samples}")
    print(f"Detected anomalies: {detected_anomalies}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

    # If ground truth labels are available, calculate metrics
    # Assuming `y_test` contains binary labels where 1 = anomaly, 0 = normal
    try:
        y_test = pd.read_csv('processed_test_labels.csv').values.flatten()
        precision = precision_score(y_test, anomalies)
        recall = recall_score(y_test, anomalies)
        f1 = f1_score(y_test, anomalies)
        auc = roc_auc_score(y_test, reconstruction_errors)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
    except FileNotFoundError:
        print("Ground truth labels not found. Skipping metrics computation.")

if __name__ == '__main__':
    main()