import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train_autoencoder(X, encoding_dim=2):
    """
    Train an autoencoder for dimensionality reduction.
    """
    input_dim = X.shape[1]

    # Define the autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile and train the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=1)

    return encoder

def train_kmeans_with_tf(X, n_clusters=5):
    """
    Train a K-Means clustering model using TensorFlow.
    """
    from tensorflow.keras.initializers import RandomUniform

    # Initialize cluster centers
    cluster_centers = tf.Variable(
        initial_value=RandomUniform()(shape=(n_clusters, X.shape[1])),
        trainable=True
    )

    # Assign points to nearest cluster
    def assign_clusters(X, cluster_centers):
        distances = tf.norm(X[:, None] - cluster_centers[None, :], axis=2)
        return tf.argmin(distances, axis=1)

    optimizer = tf.optimizers.Adam(learning_rate=0.1)

    for step in range(100):
        with tf.GradientTape() as tape:
            distances = tf.norm(X[:, None] - cluster_centers[None, :], axis=2)
            cluster_assignments = tf.argmin(distances, axis=1)

            # Compute loss as the sum of distances within clusters
            loss = tf.reduce_sum(
                tf.reduce_min(distances, axis=1)
            )

        gradients = tape.gradient(loss, [cluster_centers])
        optimizer.apply_gradients(zip(gradients, [cluster_centers]))

    cluster_assignments = assign_clusters(X, cluster_centers)
    return cluster_centers, cluster_assignments.numpy()

def plot_clusters(X, cluster_assignments):
    """
    Plot the clusters using the first two principal components.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.5)
    plt.title('K-Means Clustering (PCA-reduced Data)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()

def main():
    # Load the preprocessed dataset
    X_train = pd.read_csv('datasets/processed_data/processed_train_features.csv').values

    # Train autoencoder for dimensionality reduction
    encoder = train_autoencoder(X_train, encoding_dim=2)
    X_encoded = encoder.predict(X_train)

    # Train K-Means model with TensorFlow
    cluster_centers, cluster_assignments = train_kmeans_with_tf(X_encoded, n_clusters=5)

    # Plot clusters
    plot_clusters(X_encoded, cluster_assignments)

    # Save the cluster centers
    np.save('models/clustering/tf_kmeans_centers.npy', cluster_centers.numpy())
    print("Cluster centers saved to 'models/clustering/tf_kmeans_centers.npy'")

if __name__ == '__main__':
    main()
