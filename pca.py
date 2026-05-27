"""
Principal Component Analysis (PCA) Implementation
=================================================
PCA is a dimensionality reduction technique that:
1. Finds principal components (directions of maximum variance)
2. Projects data onto lower-dimensional subspace
3. Preserves most information with fewer features

Mathematical Concept:
- Compute covariance matrix of data
- Find eigenvectors (principal components) and eigenvalues
- Project data onto top k eigenvectors
- Variance explained by each component = eigenvalue / total_variance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) Implementation.

    Parameters:
    -----------
    n_components : int, default=2
        Number of principal components to keep
    """

    def __init__(self, n_components=2):
        """Initialize PCA parameters."""
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        Fit PCA model by computing principal components.

        Algorithm:
        Step 1: Standardize features (zero mean)
        Step 2: Compute covariance matrix
        Step 3: Compute eigenvalues and eigenvectors
        Step 4: Sort by eigenvalues (descending) and select top k

        Mathematical Steps:
        1. Center: X_centered = X - mean(X)
        2. Covariance: Cov = (X_centered)^T * X_centered / (n-1)
        3. Eigendecomposition: Cov * v = λ * v
        4. Top components = eigenvectors with largest eigenvalues

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self : fitted model
        """
        # Step 1: Standardize (center) the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 4: Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        self.components = eigenvectors[:, : self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(
            f"Cumulative explained variance: {np.sum(self.explained_variance_ratio):.4f}"
        )

        return self

    def transform(self, X):
        """
        Project data onto principal components.

        Mathematical Formula:
        X_projected = (X - mean) * components

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform

        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Projected data
        """
        if self.components is None:
            raise ValueError("Model must be fit before transform.")

        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        """Fit model and transform data in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct data from projected space.

        Parameters:
        -----------
        X_transformed : array, shape (n_samples, n_components)
            Projected data

        Returns:
        --------
        X_reconstructed : array, shape (n_samples, n_features)
            Reconstructed data
        """
        if self.components is None:
            raise ValueError("Model must be fit before inverse_transform.")

        return (X_transformed @ self.components.T) + self.mean

    def get_explained_variance_ratio(self):
        """Return explained variance ratio for each component."""
        return self.explained_variance_ratio

    def get_cumulative_explained_variance(self):
        """Return cumulative explained variance."""
        return np.cumsum(self.explained_variance_ratio)


# ============================================================================
# SAMPLE USAGE: PCA for Dimensionality Reduction
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA) - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load iris dataset
    print("\n[Step 1] Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Original features: {iris.feature_names}")

    # Step 2: Standardize features
    print("\n[Step 2] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Determine optimal number of components
    print("\n[Step 3] Determining optimal number of components...")

    pca_full = PrincipalComponentAnalysis(n_components=X.shape[1])
    pca_full.fit(X_scaled)

    cumsum_var = pca_full.get_cumulative_explained_variance()

    print(f"\nExplained Variance by Component:")
    print(f"{'Component':<12} {'Variance Ratio':<18} {'Cumulative':<15}")
    print("-" * 45)

    for i, (var, cum_var) in enumerate(
        zip(pca_full.get_explained_variance_ratio(), cumsum_var)
    ):
        print(f"{i+1:<12} {var:<18.4f} {cum_var:<15.4f}")

    # Find n_components for 95% variance
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")

    # Step 4: Fit PCA with 2 components for visualization
    print("\n[Step 4] Fitting PCA with 2 components for visualization...")
    pca = PrincipalComponentAnalysis(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 5: Analyze components
    print("\n[Step 5] Principal Components Analysis:")
    print(f"Shape of components: {pca.components.shape}")
    print(f"\\nFirst Principal Component loadings:")
    for feat_name, loading in zip(iris.feature_names, pca.components[:, 0]):
        print(f"  {feat_name}: {loading:.4f}")

    print(f"\\nSecond Principal Component loadings:")
    for feat_name, loading in zip(iris.feature_names, pca.components[:, 1]):
        print(f"  {feat_name}: {loading:.4f}")

    # Step 6: Reconstruction error
    print("\n[Step 6] Reconstruction Analysis:")
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"Mean Squared Reconstruction Error: {reconstruction_error:.6f}")

    # Step 7: Visualizations
    print("\n[Step 7] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: 2D PCA scatter
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y))))
    for cls in np.unique(y):
        mask = y == cls
        axes[0, 0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[colors[cls]],
            label=iris.target_names[cls],
            s=100,
            alpha=0.7,
            edgecolors="k",
        )

    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.2%})", fontsize=11)
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.2%})", fontsize=11)
    axes[0, 0].set_title(
        "PCA Projection (2 Components)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Scree plot (explained variance)
    axes[0, 1].plot(
        range(1, len(pca_full.explained_variance_ratio) + 1),
        pca_full.explained_variance_ratio,
        marker="o",
        linewidth=2,
        markersize=8,
    )
    axes[0, 1].set_xlabel("Principal Component", fontsize=11)
    axes[0, 1].set_ylabel("Explained Variance Ratio", fontsize=11)
    axes[0, 1].set_title("Scree Plot", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cumulative explained variance
    axes[1, 0].plot(
        range(1, len(cumsum_var) + 1),
        cumsum_var,
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    axes[1, 0].axhline(
        0.95, color="red", linestyle="--", linewidth=2, label="95% threshold"
    )
    axes[1, 0].axvline(n_components_95, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Number of Components", fontsize=11)
    axes[1, 0].set_ylabel("Cumulative Explained Variance", fontsize=11)
    axes[1, 0].set_title(
        "Cumulative Explained Variance", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])

    # Plot 4: Component loadings heatmap
    loading_matrix = pca_full.components[:n_components_95].T
    im = axes[1, 1].imshow(loading_matrix, cmap="RdBu_r", aspect="auto")
    axes[1, 1].set_xticks(range(n_components_95))
    axes[1, 1].set_xticklabels([f"PC{i+1}" for i in range(n_components_95)])
    axes[1, 1].set_yticks(range(len(iris.feature_names)))
    axes[1, 1].set_yticklabels(iris.feature_names, fontsize=10)
    axes[1, 1].set_title("PCA Component Loadings", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("pca_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'pca_visualization.png'")
    plt.show()

    # Step 8: Summary
    print("\n[Step 8] Summary:")
    print(f"Original dimensionality: {X.shape[1]}")
    print(f"Reduced dimensionality (2 components): 2")
    print(f"Variance preserved: {np.sum(pca.explained_variance_ratio):.2%}")
    print(f"Dimensionality reduction: {(1 - 2/X.shape[1])*100:.1f}%")

    print("\n" + "=" * 70)
    print("PCA completed successfully!")
    print("=" * 70)
