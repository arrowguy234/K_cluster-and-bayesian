
# Project Title: Stellar Cluster Distance Estimation using Bayesian and K-means Clustering

This project leverages custom implementations of K-means clustering and Bayesian Gaussian Mixture Models (GMM) to analyze and estimate distances to stellar clusters using data from the Gaia mission. The project includes preprocessing steps, clustering algorithms, and visualization techniques for understanding stellar cluster properties.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/arrowguy234/K_cluster-and-bayesian.git>
   cd <repository-folder>
   ```


2. Ensure you have Python 3.8 or higher installed.

---

## Usage

### 1. Data Preparation
- Place your data in the `data/` directory in CSV format.
- Ensure the dataset contains the following columns:
  - `parallax`: Parallax values (arcseconds)
  - `ra`: Right Ascension
  - `dec`: Declination
-You can run Gaia_visualisation.py file for database related plots and statistical details.

### 2. Running Clustering Algorithms

#### K-means Clustering

Run the `custom_kmeans.py` file:
```bash
python custom_kmeans.py
```

#### Bayesian Clustering (GMM)

Run the `bayesian.py` file:
```bash
python bayesian.py
```

This will:
- Add Bayesian and k means cluster assignments to the input dataset.
- Generate 3D cluster visualizations.

#### Silhouette scores

Run the `silhouttescores.py` file:
```bash
python silhouttescores.py
```

This will:
- Get functions ready for calculating silhouette scores in main file


#### Main file

Run the `main.py` file:
```bash
python main.py
```

This will:
- Print clustering plots for K means and bayesian.
- Print WCSS and log likelihood values 
- Saves values of both clustering algorithms as csv file
-Print silhouette scores
---

## Code Overview

### 1. `custom_kmeans.py`
- **`CustomKMeans` Class**:
  - Implements a basic K-means algorithm with support for cluster statistics.
  - Methods include:
    - `fit(X)`: Fit the clustering model to the data.
    - `predict(X)`: Predict clusters for new data points.
    - `get_cluster_stats()`: Retrieve cluster statistics (size, intra-cluster distances, centroids).

### 2. `bayesian.py`
- **Key Functions**:
  - `calculate_distance(parallax)`: Computes distance from parallax values.
  - `normalize_data(X)`: Scales data features to [0, 1].
  - `perform_bayesian_clustering(data, features, n_clusters, max_iter=100, tol=1e-4)`: Clusters data using GMM and visualizes results.

### 3. `main.py`
- Entry point for running the clustering algorithms.
- Parses command-line arguments and executes the appropriate clustering algorithm.

---

## Example Workflow

1. Normalize data and compute distances:
   ```python
   from bayesian import calculate_distance, normalize_data
   data['distance'] = calculate_distance(data['parallax'])
   features = normalize_data(data[['ra', 'dec', 'distance']].values)
   ```

2. Perform clustering using K-means:
   ```python
   from custom_kmeans import CustomKMeans
   kmeans = CustomKMeans(n_clusters=5, tol=1e-7)
   kmeans.fit(features)
   print(kmeans.get_cluster_stats())
   ```

3. Perform Bayesian clustering:
   ```python
   from bayesian import perform_bayesian_clustering
   perform_bayesian_clustering(data, features, n_clusters=5)
   ```

---

## Output Details

- **K-means Outputs**:
  - Cluster assignments and centroids for each cluster.
  - Intra-cluster distance and cluster sizes.

- **Bayesian Outputs**:
  - Log-likelihood of clustering.
  - 3D visualization of clusters and centroids.

---

## Dependencies

- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  


---

## Contact

For questions or issues, please contact:
- **Surinder Singh Chhabra**  
  Email: [surinder.chhabra0000@gmail.com](mailto:surinder.chhabra0000@gmail.com)
