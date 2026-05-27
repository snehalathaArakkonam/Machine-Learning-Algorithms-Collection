"""
K-Means Clustering Implementation
==================================
K-Means is an unsupervised learning algorithm that partitions data into k clusters
by iteratively assigning points to nearest centroids and updating centroids.

Algorithm:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid (Lloyd's algorithm)
3. Update centroids as mean of assigned points
4. Repeat until convergence

Convergence: When centroid positions change by less than threshold or max iterations reached.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler


class KMeans:
    """
    K-Means Clustering Algorithm.

    Parameters:
    -----------
    n_clusters : int, default=3
        Number of clusters
    max_iterations : int, default=100
        Maximum iterations for convergence
    tolerance : float, default=1e-4
        Tolerance for centroid change (convergence threshold)
    random_state : int, default=None
        Random seed for reproducibility
    init_method : str, default='random'
        Initialization method ('random', 'kmeans++')
    """

    def __init__(
        self,
        n_clusters=3,
        max_iterations=100,
        tolerance=1e-4,
        random_state=None,
        init_method="random",
    ):
        """Initialize K-Means parameters."""
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_history = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X):
        """
        Fit K-Means model by finding optimal centroids.

        Step 1: Initialize centroids
        Step 2: Iterate until convergence:
            a. Assignment: Assign each point to nearest centroid
            b. Update: Recalculate centroids as mean of assigned points
        Step 3: Calculate final inertia (within-cluster sum of squares)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster

        Returns:
        --------
        self : fitted model
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize centroids
        if self.init_method == "kmeans++":
            self.centroids = self._init_centroids_kmeans_plus(X)
        else:
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[indices].copy()

        self.inertia_history = []

        print(f"K-Means clustering with k={self.n_clusters}")

        # Step 2: Iterate until convergence
        for iteration in range(self.max_iterations):
            # Step 2a: Assign points to nearest centroid
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            # Calculate inertia (within-cluster sum of squares)
            inertia = 0
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    inertia += np.sum((cluster_points - self.centroids[k]) ** 2)

            self.inertia_history.append(inertia)

            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()

            # Step 2b: Update centroids
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # If cluster is empty, keep old centroid or reinitialize
                    self.centroids[k] = old_centroids[k]

            # Check for convergence
            centroid_shift = np.sum(
                np.linalg.norm(self.centroids - old_centroids, axis=1)
            )

            if (iteration + 1) % 10 == 0:
                print(
                    f"  Iteration {iteration + 1}: Inertia = {inertia:.4f}, Shift = {centroid_shift:.6f}"
                )

            if centroid_shift < self.tolerance:
                print(f"Converged at iteration {iteration + 1}")
                break

        return self

    def _init_centroids_kmeans_plus(self, X):
        """
        K-Means++ initialization for better initial centroids.

        Algorithm:
        1. Choose first centroid randomly from data
        2. For remaining k-1 centroids:
            - For each point, compute D(x) = distance to nearest centroid
            - Choose new centroid with probability proportional to D(x)²

        This leads to better convergence.
        """
        n_samples = X.shape[0]
        centroids = []

        # Choose first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        # Choose remaining centroids
        for _ in range(self.n_clusters - 1):
            centroids_array = np.array(centroids)
            distances = self._compute_distances(X, centroids_array)
            min_distances = np.min(distances, axis=1)

            # Probability proportional to D(x)²
            probabilities = min_distances**2
            probabilities /= np.sum(probabilities)

            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _compute_distances(self, X, centroids):
        """
        Compute Euclidean distances between points and centroids.

        Mathematical Formula:
        distance = sqrt(sum((x_i - c_j)²))

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
        centroids : array, shape (n_clusters, n_features)

        Returns:
        --------
        distances : array, shape (n_samples, n_clusters)
        """
        distances = np.zeros((X.shape[0], centroids.shape[0]))

        for i, x in enumerate(X):
            for j, c in enumerate(centroids):
                distances[i, j] = np.linalg.norm(x - c)

        return distances

    def predict(self, X):
        """
        Predict cluster labels for new samples.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster label for each sample
        """
        if self.centroids is None:
            raise ValueError("Model must be fit before predictions.")

        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Fit model and return cluster labels."""
        self.fit(X)
        return self.labels


# ============================================================================
# SAMPLE USAGE: K-Means Clustering
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("K-MEANS CLUSTERING - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create synthetic dataset with clusters
    print("\n[Step 1] Creating synthetic clustered dataset...")
    X, y_true = make_blobs(
        n_samples=300, centers=4, n_features=2, cluster_std=0.60, random_state=42
    )

    print(f"Dataset shape: {X.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")

    # Step 2: Standardize features
    print("\n[Step 2] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Test different k values using Elbow method
    print("\n[Step 3] Testing different k values (Elbow method)...")
    k_values = range(1, 8)
    inertias = []
    models = []

    for k in k_values:
        model = KMeans(n_clusters=k, max_iterations=100, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia_history[-1])
        models.append(model)
        print(f"  k={k}: Inertia = {inertias[-1]:.4f}")

    # Step 4: Choose optimal k (typically at elbow)
    optimal_k = 4
    print(f"\n[Step 4] Using optimal k={optimal_k}...")

    model = KMeans(n_clusters=optimal_k, max_iterations=100, random_state=42)
    model.fit(X_scaled)
    labels = model.labels
    centroids = model.centroids

    # Step 5: Analyze clusters
    print("\n[Step 5] Cluster Analysis:")
    print(f"{'Cluster':<10} {'Size':<10} {'Centroid':<30}")
    print("-" * 50)

    for k in range(optimal_k):
        cluster_size = np.sum(labels == k)
        centroid = centroids[k]
        print(f"{k:<10} {cluster_size:<10} {str(centroid):<30}")

    # Step 6: Calculate silhouette coefficient
    print("\n[Step 6] Cluster Quality Metrics:")

    # Silhouette coefficient
    from sklearn.metrics import silhouette_score

    silhouette = silhouette_score(X_scaled, labels)
    print(f"Silhouette Coefficient: {silhouette:.4f} (range: [-1, 1])")
    print(f"  Values closer to 1 indicate well-separated clusters")

    # Davies-Bouldin Index
    from sklearn.metrics import davies_bouldin_score

    db_index = davies_bouldin_score(X_scaled, labels)
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"  Lower values indicate better separation")

    # Step 7: Visualizations
    print("\n[Step 7] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Clustered data
    colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))

    for k in range(optimal_k):
        cluster_points = X_scaled[labels == k]
        axes[0, 0].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[k]],
            label=f"Cluster {k}",
            s=50,
            alpha=0.7,
            edgecolors="k",
        )

    # Plot centroids
    axes[0, 0].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="*",
        s=500,
        edgecolors="black",
        linewidth=2,
        label="Centroids",
    )

    axes[0, 0].set_xlabel("Feature 1", fontsize=11)
    axes[0, 0].set_ylabel("Feature 2", fontsize=11)
    axes[0, 0].set_title(
        f"K-Means Clustering (k={optimal_k})", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Elbow curve
    axes[0, 1].plot(
        k_values, inertias, marker="o", linewidth=2, markersize=8, color="steelblue"
    )
    axes[0, 1].scatter(
        [optimal_k],
        [inertias[optimal_k - 1]],
        color="red",
        s=200,
        marker="*",
        label=f"Optimal k={optimal_k}",
        zorder=5,
    )
    axes[0, 1].set_xlabel("Number of Clusters (k)", fontsize=11)
    axes[0, 1].set_ylabel("Inertia (Within-Cluster Sum)", fontsize=11)
    axes[0, 1].set_title("Elbow Method", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xticks(k_values)

    # Plot 3: Cluster sizes
    cluster_sizes = np.bincount(labels)
    bars = axes[1, 0].bar(
        range(optimal_k), cluster_sizes, color=colors, edgecolor="black", linewidth=2
    )
    axes[1, 0].set_xlabel("Cluster", fontsize=11)
    axes[1, 0].set_ylabel("Number of Points", fontsize=11)
    axes[1, 0].set_title("Cluster Sizes", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(range(optimal_k))

    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 4: Inertia evolution during training
    axes[1, 1].plot(model.inertia_history, linewidth=2, color="steelblue")
    axes[1, 1].set_xlabel("Iteration", fontsize=11)
    axes[1, 1].set_ylabel("Inertia", fontsize=11)
    axes[1, 1].set_title(
        "Inertia Reduction During Training", fontsize=12, fontweight="bold"
    )
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kmeans_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'kmeans_visualization.png'")
    plt.show()

    # Step 8: Predict on new samples
    print("\n[Step 8] Predicting cluster labels for new samples:")
    new_samples = np.array([[0, 0], [3, 3], [-3, -3]])
    new_samples_scaled = scaler.transform(new_samples)
    predictions = model.predict(new_samples_scaled)

    print(f"New samples -> Predicted clusters:")
    for sample, pred in zip(new_samples, predictions):
        print(f"  {sample} -> Cluster {pred}")

    print("\n" + "=" * 70)
    print("K-Means Clustering completed successfully!")
    print("=" * 70)
