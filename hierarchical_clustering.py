"""
Hierarchical Clustering Implementation
=======================================
Hierarchical clustering creates a tree-like hierarchy of clusters (dendrogram).
Two main approaches:
1. Agglomerative (bottom-up): Start with each point as cluster, merge closer clusters
2. Divisive (top-down): Start with one cluster, recursively split

Linkage Criteria determine distance between clusters:
- Single: min distance between any two points
- Complete: max distance between any two points
- Average: average distance between all pairs
- Ward: minimizes within-cluster variance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering Algorithm.

    Parameters:
    -----------
    linkage_criterion : str, default='ward'
        Linkage method ('single', 'complete', 'average', 'ward')
    distance_metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan')
    """

    def __init__(self, linkage_criterion="ward", distance_metric="euclidean"):
        """Initialize Hierarchical Clustering parameters."""
        self.linkage_criterion = linkage_criterion
        self.distance_metric = distance_metric
        self.clusters = None
        self.linkage_matrix = None
        self.dendrogram_data = None

    def fit(self, X):
        """
        Build hierarchical clustering tree.

        Algorithm (Agglomerative):
        Step 1: Initialize - each point is its own cluster
        Step 2: Repeat until one cluster remains:
            a. Compute distances between all pairs of clusters
            b. Merge two closest clusters
            c. Update cluster distances
        Step 3: Build linkage matrix representing merge sequence

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster

        Returns:
        --------
        self : fitted model
        """
        n_samples = X.shape[0]

        # Step 1: Initialize - each point is its own cluster
        # Clusters stored as sets of point indices
        self.clusters = [[i] for i in range(n_samples)]

        # Distance matrix - computed once for efficiency
        distances = self._compute_distance_matrix(X)

        self.linkage_matrix = []

        print(f"Hierarchical Clustering with {self.linkage_criterion} linkage")

        # Step 2: Agglomerative clustering
        cluster_id = n_samples  # For naming newly merged clusters
        iteration = 0

        while len(self.clusters) > 1:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = 0, 1

            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    # Compute distance between clusters
                    dist = self._cluster_distance(
                        X, self.clusters[i], self.clusters[j], distances
                    )

                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge closest clusters
            merged = self.clusters[merge_i] + self.clusters[merge_j]

            # Record merge in linkage matrix
            # [cluster_id_1, cluster_id_2, distance, n_samples_in_merged_cluster]
            self.linkage_matrix.append([merge_i, merge_j, min_dist, len(merged)])

            # Remove old clusters and add merged cluster
            # Remove larger index first to avoid index shifting
            if merge_i > merge_j:
                self.clusters.pop(merge_i)
                self.clusters.pop(merge_j)
            else:
                self.clusters.pop(merge_j)
                self.clusters.pop(merge_i)

            self.clusters.append(merged)

            iteration += 1
            if iteration % 50 == 0:
                print(f"  Merged clusters {iteration}, remaining: {len(self.clusters)}")

        self.linkage_matrix = np.array(self.linkage_matrix)
        print(f"✓ Clustering complete! Created {len(self.linkage_matrix)} merges")

        return self

    def _compute_distance_matrix(self, X):
        """Precompute pairwise distances between all points."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.distance_metric == "euclidean":
                    dist = np.linalg.norm(X[i] - X[j])
                elif self.distance_metric == "manhattan":
                    dist = np.sum(np.abs(X[i] - X[j]))
                else:
                    raise ValueError(f"Unknown distance metric: {self.distance_metric}")

                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _cluster_distance(self, X, cluster1, cluster2, distances):
        """
        Compute distance between two clusters based on linkage criterion.

        Linkage Criteria:
        - Single: min distance between any two points
        - Complete: max distance between any two points
        - Average: average distance between all pairs
        - Ward: minimizes within-cluster variance increase
        """
        if self.linkage_criterion == "single":
            # Minimum distance (single linkage)
            min_dist = np.inf
            for i in cluster1:
                for j in cluster2:
                    min_dist = min(min_dist, distances[i, j])
            return min_dist

        elif self.linkage_criterion == "complete":
            # Maximum distance (complete linkage)
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    max_dist = max(max_dist, distances[i, j])
            return max_dist

        elif self.linkage_criterion == "average":
            # Average distance
            total_dist = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += distances[i, j]
            return total_dist / (len(cluster1) * len(cluster2))

        elif self.linkage_criterion == "ward":
            # Ward's method (minimize within-cluster variance)
            cluster1_center = X[cluster1].mean(axis=0)
            cluster2_center = X[cluster2].mean(axis=0)
            merged_center = X[cluster1 + cluster2].mean(axis=0)

            # Within-cluster variance increase
            var1 = np.sum((X[cluster1] - cluster1_center) ** 2)
            var2 = np.sum((X[cluster2] - cluster2_center) ** 2)
            var_merged = np.sum((X[cluster1 + cluster2] - merged_center) ** 2)

            return var_merged - (var1 + var2)

        else:
            raise ValueError(f"Unknown linkage criterion: {self.linkage_criterion}")

    def get_flat_clusters(self, n_clusters):
        """
        Cut dendrogram at height to get flat clustering.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters to extract

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster label for each sample
        """
        # This is a simplified version
        # Full implementation would use linkage matrix properly
        if self.linkage_matrix is None:
            raise ValueError("Model must be fit before getting clusters.")

        # For now, return random assignment
        # In production, implement proper dendrogram cutting
        print(f"Extracting {n_clusters} clusters from hierarchy")
        return np.array(
            [i % n_clusters for i in range(self.linkage_matrix.shape[0] + 1)]
        )


# ============================================================================
# SAMPLE USAGE: Hierarchical Clustering with scipy
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL CLUSTERING - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create synthetic dataset
    print("\n[Step 1] Creating synthetic dataset...")
    X, y_true = make_blobs(
        n_samples=80, centers=4, n_features=2, cluster_std=0.8, random_state=42
    )

    print(f"Dataset shape: {X.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")

    # Step 2: Standardize features
    print("\n[Step 2] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Test different linkage methods
    print("\n[Step 3] Testing different linkage methods...")
    linkage_methods = ["single", "complete", "average", "ward"]
    linkage_matrices = {}

    for method in linkage_methods:
        print(f"  Computing linkage with {method} method...")
        Z = linkage(X_scaled, method=method)
        linkage_matrices[method] = Z

    # Step 4: Visualizations
    print("\n[Step 4] Generating dendrogram visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, method in enumerate(linkage_methods):
        print(f"  Plotting dendrogram for {method} linkage...")

        dendrogram(linkage_matrices[method], ax=axes[idx], leaf_font_size=10)
        axes[idx].set_title(
            f"Dendrogram ({method} linkage)", fontsize=12, fontweight="bold"
        )
        axes[idx].set_xlabel("Sample Index", fontsize=11)
        axes[idx].set_ylabel("Distance", fontsize=11)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hierarchical_clustering_dendrograms.png", dpi=300, bbox_inches="tight")
    print("✓ Dendrograms saved as 'hierarchical_clustering_dendrograms.png'")
    plt.show()

    # Step 5: Extract clusters from dendrogram
    print("\n[Step 5] Extracting clusters from dendrogram...")
    from scipy.cluster.hierarchy import fcluster

    # Use Ward linkage (usually best)
    Z = linkage_matrices["ward"]

    # Extract 4 clusters
    n_clusters = 4
    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    print(f"Extracted {n_clusters} clusters from dendrogram")
    print(f"Cluster sizes: {np.bincount(labels)}")

    # Step 6: Cluster quality metrics
    print("\n[Step 6] Cluster Quality Metrics:")

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)

    print(f"Silhouette Coefficient: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    # Step 7: Scatter plot of clusters
    print("\n[Step 7] Generating scatter plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: True clusters
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y_true))))
    for cls in np.unique(y_true):
        mask = y_true == cls
        axes[0].scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            c=[colors[cls]],
            label=f"Cluster {cls}",
            s=100,
            alpha=0.7,
            edgecolors="k",
        )

    axes[0].set_xlabel("Feature 1", fontsize=11)
    axes[0].set_ylabel("Feature 2", fontsize=11)
    axes[0].set_title("True Clusters", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Hierarchical clustering results
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    for cls in range(n_clusters):
        mask = labels == cls
        axes[1].scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            c=[colors[cls]],
            label=f"Cluster {cls}",
            s=100,
            alpha=0.7,
            edgecolors="k",
        )

    axes[1].set_xlabel("Feature 1", fontsize=11)
    axes[1].set_ylabel("Feature 2", fontsize=11)
    axes[1].set_title(
        f"Hierarchical Clustering (Ward, {n_clusters} clusters)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hierarchical_clustering_scatter.png", dpi=300, bbox_inches="tight")
    print("✓ Scatter plot saved as 'hierarchical_clustering_scatter.png'")
    plt.show()

    # Step 8: Compare with true labels (if applicable)
    print("\n[Step 8] Comparison with True Labels:")
    print(
        f"Adjusted Rand Index: {adjusted_rand_index(y_true, labels):.4f}"
        if "adjusted_rand_index" in dir()
        else "True labels provided for reference"
    )

    print("\n" + "=" * 70)
    print("Hierarchical Clustering completed successfully!")
    print("=" * 70)


def adjusted_rand_index(y_true, y_pred):
    """Simple implementation of Adjusted Rand Index."""
    from sklearn.metrics import adjusted_rand_score

    return adjusted_rand_score(y_true, y_pred)
