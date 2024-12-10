import numpy as np  # Importing numpy for numerical operations

def calculate_distance(parallax):
    """Calculate distances in parsecs using parallax in milliarcseconds."""
    # The formula to calculate distance in parsecs from parallax (in milliarcseconds) is:
    # distance (pc) = 1000 / parallax (mas)
    return 1000 / parallax  # Return the calculated distance in parsecs

def calculate_silhouette_per_cluster(features, labels):
    """
    Calculate Silhouette Score for each cluster.
    
    The Silhouette Score measures how similar an object is to its own cluster 
    (cohesion) compared to other clusters (separation). A higher score indicates
    better clustering.
    
    Parameters:
    - features: numpy array of shape (n_samples, n_features), the data points' feature vectors.
    - labels: numpy array of shape (n_samples,), the cluster labels assigned to each data point.
    
    Returns:
    - cluster_scores: dictionary where keys are the unique cluster labels and 
      values are the average Silhouette Scores for each cluster.
    """
    # Get unique cluster labels from the 'labels' array
    unique_labels = np.unique(labels)
    
    # Dictionary to store Silhouette Scores for each cluster
    cluster_scores = {}

    # Loop over each unique cluster label
    for cluster_label in unique_labels:
        # Find the indices of the data points belonging to the current cluster
        cluster_indices = np.where(labels == cluster_label)[0]
        
        # List to hold silhouette scores for each data point in the current cluster
        cluster_silhouette_scores = []

        # Loop over each data point in the current cluster
        for i in cluster_indices:
            # Get all points that belong to the same cluster as the current point
            same_cluster = features[labels == labels[i]]
            
            # Calculate intra-cluster distances (distance between the point and others in the same cluster)
            intra_distances = np.linalg.norm(same_cluster - features[i], axis=1)
            
            # Compute the average distance to points in the same cluster (excluding itself)
            a_i = np.mean(intra_distances[intra_distances > 0]) if len(intra_distances) > 1 else 0

            # Initialize the minimum inter-cluster distance to infinity
            b_i = float('inf')
            
            # Loop over all other clusters to find the nearest cluster
            for other_label in unique_labels:
                if other_label != labels[i]:
                    # Get all points belonging to the other cluster
                    other_cluster = features[labels == other_label]
                    
                    # Calculate inter-cluster distances (distance between the point and points in the other cluster)
                    inter_distances = np.linalg.norm(other_cluster - features[i], axis=1)
                    
                    # Find the minimum distance to any point in the other clusters
                    b_i = min(b_i, np.mean(inter_distances))

            # Calculate the silhouette score for the current point i using the formula:
            # Silhouette Score (s_i) = (b_i - a_i) / max(a_i, b_i)
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0  # If both a_i and b_i are zero, the silhouette score is set to zero

            # Append the silhouette score for the current point i
            cluster_silhouette_scores.append(s_i)

        # Calculate and store the average silhouette score for the entire cluster
        cluster_scores[cluster_label] = np.mean(cluster_silhouette_scores)

    # Return a dictionary with the average silhouette score for each cluster
    return cluster_scores
