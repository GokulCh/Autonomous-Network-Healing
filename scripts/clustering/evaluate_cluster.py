from sklearn.decomposition import PCA

def reduce_dimensions(X, n_components):
    """
    Reduce the dimensionality of the data using PCA.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def assign_clusters(X, cluster_centers):
    """
    Assign data points to the nearest cluster center.
    """
    distances = tf.norm(X[:, None] - cluster_centers[None, :], axis=2)
    return tf.argmin(distances, axis=1)

def main():
    # Load the preprocessed test dataset
    X_test = pd.read_csv('processed_test_features.csv').values
    print(f"Original test data shape: {X_test.shape}")

    # Adjust the shape if necessary
    expected_features = 193  # As per the encoder model
    if X_test.shape[1] > expected_features:
        X_test = X_test[:, :expected_features]  # Truncate to expected features
        print(f"Truncated test data shape: {X_test.shape}")
    elif X_test.shape[1] < expected_features:
        padding = expected_features - X_test.shape[1]
        X_test = np.pad(X_test, ((0, 0), (0, padding)), 'constant')  # Pad with zeros
        print(f"Padded test data shape: {X_test.shape}")

    # Load the pre-trained encoder
    encoder_path = 'network_anomaly_autoencoder_v2.keras'
    encoder = load_encoder(encoder_path)

    # Encode the test data
    X_encoded_test = encoder.predict(X_test)
    print(f"Encoded test data shape: {X_encoded_test.shape}")

    # Reduce dimensions of the encoded data to match cluster centers
    cluster_centers_path = 'tf_kmeans_centers.npy'
    cluster_centers = load_cluster_centers(cluster_centers_path)
    n_cluster_dims = cluster_centers.shape[1]  # Dimension of cluster centers
    X_reduced_test = reduce_dimensions(X_encoded_test, n_components=n_cluster_dims)
    print(f"Reduced test data shape: {X_reduced_test.shape}")

    # Assign clusters to the test data
    cluster_assignments_test = assign_clusters(X_reduced_test, cluster_centers).numpy()

    # Plot clusters for test data
    print("Plotting test data clusters:")
    plot_clusters(X_reduced_test, cluster_assignments_test)

if __name__ == '__main__':
    main()
