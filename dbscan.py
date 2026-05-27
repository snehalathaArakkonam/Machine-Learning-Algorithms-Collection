"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
====================================================================
DBSCAN is a density-based clustering algorithm that groups points that are closely
packed together, marking outliers as noise (unlike K-Means and Hierarchical).

Key Advantages:
1. No need to specify number of clusters beforehand
2. Can find clusters of arbitrary shape
3. Identifies outliers/noise points
4. Robust to high-dimensional data

Core Concepts:
- eps: Maximum distance between two samples for them to be in same neighborhood
- min_samples: Minimum points in eps-neighborhood to form core point
- Core Point: Has at least min_samples neighbors within eps
- Border Point: Not core, but within eps of core point
- Noise Point: Neither core nor border point
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler


class DBSCAN:
    """
    DBSCAN Clustering Algorithm.

    Parameters:
    -----------
    eps : float, default=0.5
        Maximum distance between two samples for them to be neighbors
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for a point to be core point
    distance_metric : str, default='euclidean'
        Distance metric to use
    """

    def __init__(self, eps=0.5, min_samples=5, distance_metric="euclidean"):
        """Initialize DBSCAN parameters."""
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.labels = None
        self.n_clusters = None

    def fit(self, X):
        """
        Perform DBSCAN clustering.

        Algorithm:
        Step 1: For each unvisited point p:
            a. Mark p as visited
            b. Find all neighbors of p within eps distance
            c. If p is not core point, mark as noise/border
            d. If p is core point, create new cluster and expand it
        Step 2: Expand cluster by recursively adding core points

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster

        Returns:
        --------
        self : fitted model
        """
        n_samples = X.shape[0]

        # Initialize labels (-1 = noise, 0+ = cluster id)
        self.labels = np.full(n_samples, -1)

        # Precompute distance matrix for efficiency
        distances = self._compute_distances(X)

        print(f"DBSCAN with eps={self.eps}, min_samples={self.min_samples}")

        cluster_id = 0

        # Step 1: For each point
        for i in range(n_samples):
            # Skip if already assigned
            if self.labels[i] != -1:
                continue

            # Find neighbors of point i
            neighbors = self._get_neighbors(i, distances)

            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Noise point
                continue

            # Expand cluster from core point i
            self._expand_cluster(i, cluster_id, distances, neighbors)
            cluster_id += 1

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        print(f"✓ Clustering complete!")
        print(f"  Clusters found: {self.n_clusters}")
        print(f"  Noise points: {np.sum(self.labels == -1)}")

        return self

    def _compute_distances(self, X):
        """Compute pairwise distances between all points."""
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

    def _get_neighbors(self, point_idx, distances):
        """
        Get all neighbors of a point within eps distance.

        Returns indices of neighbors (including the point itself).
        """
        neighbors = np.where(distances[point_idx] <= self.eps)[0]
        return neighbors

    def _expand_cluster(self, core_idx, cluster_id, distances, neighbors):
        """
        Expand cluster from a core point using BFS.

        Recursively add all core points reachable from this core point.
        """
        # Assign core point to cluster
        self.labels[core_idx] = cluster_id

        # Use queue for BFS expansion
        queue = list(neighbors)

        while queue:
            idx = queue.pop(0)

            # Already assigned to this cluster
            if self.labels[idx] == cluster_id:
                continue

            # Mark as noise initially
            if self.labels[idx] == -1:
                self.labels[idx] = cluster_id

                # Check if this is core point
                new_neighbors = self._get_neighbors(idx, distances)
                if len(new_neighbors) >= self.min_samples:
                    # Add new neighbors to queue
                    for neighbor in new_neighbors:
                        if self.labels[neighbor] == -1:
                            queue.append(neighbor)

    def fit_predict(self, X):
        """Fit model and return cluster labels."""
        self.fit(X)
        return self.labels


# ============================================================================
# SAMPLE USAGE: DBSCAN Clustering
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DBSCAN CLUSTERING - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create datasets with different shapes
    print("\n[Step 1] Creating datasets with different shapes...")

    # Blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=300, centers=4, n_features=2, cluster_std=0.6, random_state=42
    )

    # Moons (non-convex clusters)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

    print(f"Blobs dataset shape: {X_blobs.shape}")
    print(f"Moons dataset shape: {X_moons.shape}")

    # Step 2: Standardize features
    print("\n[Step 2] Standardizing features...")
    scaler_blobs = StandardScaler()
    X_blobs_scaled = scaler_blobs.fit_transform(X_blobs)

    scaler_moons = StandardScaler()
    X_moons_scaled = scaler_moons.fit_transform(X_moons)

    # Step 3: Test different eps values
    print("\n[Step 3] Testing different eps values...")
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.7]

    results = {}
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=5)
        model.fit(X_blobs_scaled)
        n_clusters = model.n_clusters
        n_noise = np.sum(model.labels == -1)
        results[eps] = (n_clusters, n_noise)
        print(f"  eps={eps}: Clusters={n_clusters}, Noise points={n_noise}")

    # Step 4: Use optimal eps
    optimal_eps = 0.4
    print(f"\n[Step 4] Using optimal eps={optimal_eps}...")

    model_blobs = DBSCAN(eps=optimal_eps, min_samples=5)
    labels_blobs = model_blobs.fit_predict(X_blobs_scaled)

    model_moons = DBSCAN(eps=0.15, min_samples=5)
    labels_moons = model_moons.fit_predict(X_moons_scaled)

    # Step 5: Analysis
    print("\n[Step 5] Cluster Analysis:")

    unique_labels_blobs = set(labels_blobs)
    print(f"\nBlobs dataset:")
    print(f"  Number of clusters: {model_blobs.n_clusters}")
    print(f"  Noise points: {np.sum(labels_blobs == -1)}")

    for cls in sorted(unique_labels_blobs):
        if cls == -1:
            print(f"  Noise: {np.sum(labels_blobs == -1)} points")
        else:
            print(f"  Cluster {cls}: {np.sum(labels_blobs == cls)} points")

    # Step 6: Visualizations
    print("\n[Step 6] Generating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Blobs - original
    axes[0, 0].scatter(
        X_blobs[:, 0],
        X_blobs[:, 1],
        c=y_blobs,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
    )
    axes[0, 0].set_title("Blobs Dataset (True Labels)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Feature 1", fontsize=11)
    axes[0, 0].set_ylabel("Feature 2", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Blobs - DBSCAN clustering
    colors = plt.cm.Spectral(np.linspace(0, 1, model_blobs.n_clusters + 1))

    for cls in sorted(unique_labels_blobs):
        if cls == -1:
            # Noise points in black
            mask = labels_blobs == -1
            axes[0, 1].scatter(
                X_blobs_scaled[mask, 0],
                X_blobs_scaled[mask, 1],
                c="black",
                marker="x",
                s=100,
                label="Noise",
            )
        else:
            mask = labels_blobs == cls
            axes[0, 1].scatter(
                X_blobs_scaled[mask, 0],
                X_blobs_scaled[mask, 1],
                c=[colors[cls]],
                label=f"Cluster {cls}",
                s=50,
                alpha=0.7,
                edgecolors="k",
            )

    axes[0, 1].set_title(
        f"DBSCAN Blobs (eps={optimal_eps})", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Feature 1", fontsize=11)
    axes[0, 1].set_ylabel("Feature 2", fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Eps vs Clusters
    eps_list = list(results.keys())
    n_clusters_list = [results[e][0] for e in eps_list]
    n_noise_list = [results[e][1] for e in eps_list]

    axes[0, 2].plot(
        eps_list,
        n_clusters_list,
        marker="o",
        label="Clusters",
        linewidth=2,
        markersize=8,
    )
    axes[0, 2].plot(
        eps_list, n_noise_list, marker="s", label="Noise", linewidth=2, markersize=8
    )
    axes[0, 2].axvline(
        optimal_eps, color="red", linestyle="--", label=f"Selected eps={optimal_eps}"
    )
    axes[0, 2].set_xlabel("eps Parameter", fontsize=11)
    axes[0, 2].set_ylabel("Count", fontsize=11)
    axes[0, 2].set_title("Effect of eps Parameter", fontsize=12, fontweight="bold")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Moons - original
    axes[1, 0].scatter(
        X_moons[:, 0],
        X_moons[:, 1],
        c=y_moons,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
    )
    axes[1, 0].set_title("Moons Dataset (True Labels)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Feature 1", fontsize=11)
    axes[1, 0].set_ylabel("Feature 2", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Moons - DBSCAN clustering
    unique_labels_moons = set(labels_moons)
    colors_moons = plt.cm.Spectral(np.linspace(0, 1, model_moons.n_clusters + 1))

    for cls in sorted(unique_labels_moons):
        if cls == -1:
            mask = labels_moons == -1
            axes[1, 1].scatter(
                X_moons_scaled[mask, 0],
                X_moons_scaled[mask, 1],
                c="black",
                marker="x",
                s=100,
                label="Noise",
            )
        else:
            mask = labels_moons == cls
            axes[1, 1].scatter(
                X_moons_scaled[mask, 0],
                X_moons_scaled[mask, 1],
                c=[colors_moons[cls]],
                label=f"Cluster {cls}",
                s=50,
                alpha=0.7,
                edgecolors="k",
            )

    axes[1, 1].set_title(
        "DBSCAN Moons (Non-Convex Clusters)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Feature 1", fontsize=11)
    axes[1, 1].set_ylabel("Feature 2", fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Cluster sizes comparison
    cluster_sizes = np.bincount(labels_blobs[labels_blobs >= 0])
    axes[1, 2].bar(
        range(len(cluster_sizes)),
        cluster_sizes,
        color="steelblue",
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 2].set_xlabel("Cluster ID", fontsize=11)
    axes[1, 2].set_ylabel("Number of Points", fontsize=11)
    axes[1, 2].set_title("Cluster Sizes (Blobs)", fontsize=12, fontweight="bold")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("dbscan_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'dbscan_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("DBSCAN Clustering completed successfully!")
    print("=" * 70)
