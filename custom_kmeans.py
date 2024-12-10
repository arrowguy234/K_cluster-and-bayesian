import numpy as np

class CustomKMeans:
    # Initialize the KMeans algorithm with parameters:
    # - n_clusters: number of clusters (default is 5)
    # - max_iter: maximum number of iterations to run (default is 300)
    # - tol: tolerance for convergence (default is 1e-7)
    def __init__(self, n_clusters=5, max_iter=300, tol=1e-7):
        self.n_clusters = n_clusters  # Set the number of clusters
        self.max_iter = max_iter  # Set the maximum number of iterations
        self.tol = tol  # Set the tolerance for centroid convergence

    # Fit the model to the data X
    def fit(self, X):
        np.random.seed(42)  # Set the random seed for reproducibility
        # Step 1: Randomly initialize centroids by selecting random points from the data X
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]  # Initialize centroids to these random points

        # Iterate until the algorithm converges or the maximum number of iterations is reached
        for iteration in range(self.max_iter):
            # Step 2: Assign each point in X to the closest centroid
            self.labels = self._assign_clusters(X)

            # Step 3: Recalculate the centroids as the mean of the assigned points
            # If a cluster has no points assigned, retain the current centroid
            new_centroids = np.array([X[self.labels == k].mean(axis=0) if len(X[self.labels == k]) > 0 else self.centroids[k]
                                       for k in range(self.n_clusters)])

            # Step 4: Check for convergence by comparing old and new centroids
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break  # If centroids haven't changed significantly, break the loop

            # Update centroids for the next iteration
            self.centroids = new_centroids

        # After fitting, calculate cluster statistics for further analysis
        self.cluster_stats = self._calculate_cluster_stats(X)

    # Helper function to assign each point in X to the nearest centroid
    def _assign_clusters(self, X):
        # Calculate Euclidean distance from each point to each centroid
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        # For each point, assign it to the cluster corresponding to the closest centroid
        return np.argmin(distances, axis=1)

    # Predict the cluster labels for new data points
    def predict(self, X):
        # Calculate distances from each point in X to each centroid
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        # For each point, predict the label of the closest centroid (i.e., cluster)
        return np.argmin(distances, axis=1)

    # Helper function to calculate and return cluster statistics:
    # - Size of each cluster (number of points in the cluster)
    # - Intra-cluster distance (average distance of points to their centroid)
    # - Centroid coordinates for each cluster
    def _calculate_cluster_stats(self, X):
        stats = {}  # Dictionary to store statistics for each cluster
        for k in range(self.n_clusters):
            # Get the points assigned to the k-th cluster
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                # Calculate the distance of each point in the cluster from the centroid
                intra_distances = np.linalg.norm(cluster_points - self.centroids[k], axis=1)
                # Store statistics: size, mean intra-cluster distance, and centroid coordinates
                stats[k] = {
                    "size": len(cluster_points),  # Number of points in the cluster
                    "mean_distance_to_centroid": intra_distances.mean(),  # Average distance to centroid
                    "centroid": self.centroids[k]  # Coordinates of the centroid
                }
            else:
                # If no points are assigned to the cluster, set stats to default values
                stats[k] = {
                    "size": 0,  # Cluster has no points
                    "mean_distance_to_centroid": np.nan,  # No distance to centroid (empty cluster)
                    "centroid": self.centroids[k]  # Retain the centroid coordinates
                }
        return stats  # Return the cluster statistics

    # Return the cluster statistics (size, intra-cluster distances, centroids)
    def get_cluster_stats(self):
        return self.cluster_stats  # Return the computed statistics
