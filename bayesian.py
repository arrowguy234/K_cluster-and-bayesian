import numpy as np

# Normalize data without sklearn
def normalize_data(features):
    """
    Normalize the given dataset using min-max normalization.
    
    Args:
    features (numpy.ndarray): A 2D array of shape (n_samples, n_features) where each row is a data point
                              and each column represents a feature.
    
    Returns:
    numpy.ndarray: A normalized version of the input features with values scaled to the range [0, 1].
    """
    min_vals = np.min(features, axis=0)  # Find the minimum value for each feature (column)
    max_vals = np.max(features, axis=0)  # Find the maximum value for each feature (column)
    # Apply min-max scaling to normalize the data
    return (features - min_vals) / (max_vals - min_vals)  # Normalize each feature to the [0, 1] range

# Perform Bayesian clustering
def perform_bayesian_clustering(features, n_clusters=3, tolerance=1e-6, max_iter=100):
    """
    Perform clustering using a basic version of the Expectation-Maximization (EM) algorithm.

    Args:
    features (numpy.ndarray): A 2D array of shape (n_samples, n_features) representing the data to cluster.
    n_clusters (int): The number of clusters to form. Default is 3.
    tolerance (float): Convergence threshold based on centroid movement. Default is 1e-6.
    max_iter (int): Maximum number of iterations for the algorithm. Default is 100.

    Returns:
    tuple: A tuple containing:
        - centroids (numpy.ndarray): The final centroids of the clusters.
        - labels (numpy.ndarray): An array of shape (n_samples,) containing the cluster label for each sample.
    """
    # Randomly initialize centroids by selecting n_clusters random points from the dataset
    centroids = features[np.random.choice(features.shape[0], n_clusters, replace=False)]
    prev_centroids = np.zeros_like(centroids)  # Initialize previous centroids for convergence check

    labels = np.zeros(features.shape[0])  # Initialize labels (each sample is assigned a cluster)
    iteration = 0  # Track the number of iterations

    while np.linalg.norm(centroids - prev_centroids) > tolerance and iteration < max_iter:
        """
        The Expectation-Maximization (EM) algorithm works in two steps:
        1. E-step: Assign each data point to the nearest centroid.
        2. M-step: Update the centroids based on the mean of points assigned to each cluster.
        """
        # E-step: Compute distances from each point to each centroid and assign labels
        distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)  # Calculate Euclidean distance
        labels = np.argmin(distances, axis=1)  # Assign each point to the nearest centroid

        # M-step: Recompute the centroids based on the current labels (cluster assignments)
        prev_centroids = centroids.copy()  # Store the old centroids for convergence check
        for i in range(n_clusters):
            cluster_points = features[labels == i]  # Get all points assigned to cluster i
            if len(cluster_points) > 0:  # Avoid empty clusters
                centroids[i] = np.mean(cluster_points, axis=0)  # Compute new centroid as mean of points in the cluster

        iteration += 1  # Increment the iteration counter

    return centroids, labels  # Return the final centroids and the assigned labels

# Calculate log-likelihood of a clustering
def calculate_log_likelihood(features, centroids, labels, n_clusters):
    """
    Calculate the log-likelihood of a clustering based on a Gaussian Mixture Model (GMM).

    Args:
    features (numpy.ndarray): The dataset to evaluate, shape (n_samples, n_features).
    centroids (numpy.ndarray): The cluster centroids, shape (n_clusters, n_features).
    labels (numpy.ndarray): The labels of the data points, shape (n_samples,).
    n_clusters (int): The number of clusters.

    Returns:
    float: The log-likelihood value for the clustering configuration.
    """
    log_likelihood = 0  # Initialize log-likelihood to 0
    d = features.shape[1]  # The dimensionality of the data (number of features)

    # Iterate over each cluster
    for i in range(n_clusters):
        cluster_points = features[labels == i]  # Get the points assigned to cluster i
        if len(cluster_points) > 0:
            # Compute the covariance matrix of the points in the cluster, adding a small regularization value
            cov_matrix = np.cov(cluster_points.T) + np.eye(d) * 1e-6  # Regularize covariance to avoid singular matrices
            mean = centroids[i]  # The centroid (mean) of the cluster

            # Safeguard against a singular covariance matrix (non-invertible)
            try:
                cov_inv = np.linalg.inv(cov_matrix)  # Compute the inverse of the covariance matrix
                cov_det = np.linalg.det(cov_matrix)  # Compute the determinant of the covariance matrix
            except np.linalg.LinAlgError:
                continue  # Skip the current cluster if its covariance matrix is singular (non-invertible)

            # Calculate the likelihood of each point in the cluster based on a Gaussian distribution
            for point in cluster_points:
                diff = point - mean  # Difference from the point to the cluster mean (centroid)
                # Compute the likelihood using the Gaussian PDF formula
                likelihood = np.exp(-0.5 * np.dot(np.dot(diff, cov_inv), diff.T)) / \
                             np.sqrt((2 * np.pi) ** d * cov_det)

                # Avoid taking the log of zero (log(0) is undefined)
                if likelihood > 0:
                    log_likelihood += np.log(likelihood)  # Add the log of the likelihood to the total log-likelihood

    return log_likelihood  # Return the final log-likelihood value
