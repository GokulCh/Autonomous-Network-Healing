import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
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
        
        # Deep encoder with regularization
        encoded = Dense(64, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
        encoded = Dropout(0.2)(encoded)
        
        encoded = Dense(32, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoded)
        encoded = Dropout(0.2)(encoded)
        
        # Bottleneck layer
        encoded = Dense(self.encoding_dim, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoded)
        decoded = Dropout(0.2)(decoded)
        
        decoded = Dense(64, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(decoded)
        decoded = Dropout(0.2)(decoded)
        
        # Output layer reconstructs the input
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Create autoencoder model
        autoencoder = Model(input_layer, decoded)
        
        # Compile with mean squared error loss
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                            loss='mean_squared_error')
        
        return autoencoder
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32):
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
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        # Prepare validation data
        validation_data = X_val if X_val is not None else None
        
        # Train the autoencoder
        history = self.autoencoder.fit(
            X_train, X_train,  # Input is both the data and the target
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(validation_data, validation_data) if validation_data is not None else None,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history
    
    def detect_anomalies(self, X, threshold_percentile=95):
        """
        Detect anomalies using reconstruction error
        
        Parameters:
        - X (numpy.ndarray): Input data
        - threshold_percentile (float): Percentile to set anomaly threshold
        
        Returns:
        - numpy.ndarray: Boolean mask of anomalies
        """
        # Reconstruct input
        reconstructed = self.autoencoder.predict(X)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        # Determine threshold
        threshold = np.percentile(mse, threshold_percentile)
        
        # Identify anomalies
        return mse > threshold, mse

def main():
    # Load preprocessed data
    X_train = pd.read_csv('processed_train_features.csv').values
    
    # Split into train and validation
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    
    # Initialize and train autoencoder
    autoencoder = NetworkAnomalyAutoencoder(input_dim=X_train.shape[1])
    
    # Train the model
    history = autoencoder.train(X_train, X_val)
    
    # Detect anomalies in training data
    anomalies, reconstruction_errors = autoencoder.detect_anomalies(X_train)
    
    # Print anomaly detection results
    print(f"Total samples: {len(X_train)}")
    print(f"Detected anomalies: {np.sum(anomalies)}")
    print(f"Anomaly percentage: {np.mean(anomalies) * 100:.2f}%")
    
    # Save the model
    autoencoder.autoencoder.save('network_anomaly_autoencoder.h5')

if __name__ == '__main__':
    main()