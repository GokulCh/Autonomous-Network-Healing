import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NetworkAnomalyAutoencoder:
    def __init__(self, input_dim, encoding_dim=32):
        """
        Initialize Autoencoder for Network Anomaly Detection
        
        Parameters:
        - input_dim (int): Number of input features
        - encoding_dim (int): Dimensionality of the compressed representation
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = self.build_autoencoder()
    
    def build_autoencoder(self):
        """
        Build a deep autoencoder architecture for network anomaly detection
        
        Returns:
        - Compiled Keras Model
        """
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(128)(input_layer)
        encoded = LeakyReLU(alpha=0.1)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.3)(encoded)

        encoded = Dense(64)(encoded)
        encoded = LeakyReLU(alpha=0.1)(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.3)(encoded)

        # Bottleneck layer
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(64)(encoded)
        decoded = LeakyReLU(alpha=0.1)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.3)(decoded)

        decoded = Dense(128)(decoded)
        decoded = LeakyReLU(alpha=0.1)(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.3)(decoded)

        # Output layer reconstructs the input
        output_layer = Dense(self.input_dim, activation='linear')(decoded)

        # Create autoencoder model
        autoencoder = Model(input_layer, output_layer)

        # Compile with mean squared error loss
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
            loss='mean_squared_error'
        )
        
        return autoencoder
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=64):
        """
        Train the autoencoder
        
        Parameters:
        - X_train (numpy.ndarray): Training data
        - X_val (numpy.ndarray, optional): Validation data
        - epochs (int): Number of training epochs
        - batch_size (int): Batch size for training
        """
        # Prepare callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train the autoencoder
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, X_val) if X_val is not None else None,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def detect_anomalies(self, X, threshold_factor=1.5):
        """
        Detect anomalies using reconstruction error
        
        Parameters:
        - X (numpy.ndarray): Input data
        - threshold_factor (float): Multiplier for mean reconstruction error to set anomaly threshold
        
        Returns:
        - numpy.ndarray: Boolean mask of anomalies
        - numpy.ndarray: Reconstruction error values
        """
        # Reconstruct input
        reconstructed = self.autoencoder.predict(X)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        # Adaptive threshold
        threshold = mse.mean() + threshold_factor * mse.std()
        
        # Identify anomalies
        return mse > threshold, mse

    def plot_training_history(self, history):
        """
        Plot training and validation loss over epochs.
        
        Parameters:
        - history: Keras training history object
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_anomaly_distribution(self, reconstruction_errors, threshold):
        """
        Plot the distribution of reconstruction errors with the anomaly threshold.
        
        Parameters:
        - reconstruction_errors (numpy.ndarray): Reconstruction errors
        - threshold (float): Anomaly detection threshold
        """
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_errors, bins=50, color='blue', alpha=0.7, label='Reconstruction Errors')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.show()

def main():
    # Load preprocessed data
    X_train = pd.read_csv('processed_train_features.csv').values

    # Split into train and validation
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Initialize and train autoencoder
    autoencoder = NetworkAnomalyAutoencoder(input_dim=X_train.shape[1])

    # Train the model
    history = autoencoder.train(X_train, X_val)

    # Plot training history
    autoencoder.plot_training_history(history)

    # Detect anomalies in training data
    anomalies, reconstruction_errors = autoencoder.detect_anomalies(X_train)
    
    # Calculate adaptive threshold
    threshold = reconstruction_errors.mean() + 1.5 * reconstruction_errors.std()

    # Plot anomaly distribution
    autoencoder.plot_anomaly_distribution(reconstruction_errors, threshold)

    # Print anomaly detection results
    print(f"Total samples: {len(X_train)}")
    print(f"Detected anomalies: {np.sum(anomalies)}")
    print(f"Anomaly percentage: {np.mean(anomalies) * 100:.2f}%")

    # Save the model
    autoencoder.autoencoder.save('network_anomaly_autoencoder_v2.h5')

if __name__ == '__main__':
    main()