import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def train_kmeans(X, n_clusters=5):
    """
    Train a K-Means clustering model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    
    return kmeans, silhouette_avg

def plot_clusters(X, kmeans):
    """
    Plot the clusters using the first two principal components
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    plt.title('K-Means Clustering (PCA-reduced Data)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()

def main():
    # Load the preprocessed dataset
    X_train = pd.read_csv('datasets/processed_data/processed_train_features.csv')
    
    # Train K-Means model
    kmeans, silhouette_avg = train_kmeans(X_train.values, n_clusters=5)
    
    # Print silhouette score
    print(f'Silhouette Score: {silhouette_avg:.4f}')
    
    # Plot clusters
    plot_clusters(X_train.values, kmeans)
    
    # Save the model
    import joblib
    joblib.dump(kmeans, 'models/clustering/kmeans_model.pkl')
    print("K-Means model saved to 'models/clustering/kmeans_model.pkl'")

if __name__ == '__main__':
    main()