"""
Linear Discriminant Analysis (LDA) Implementation
=================================================
LDA is both a dimensionality reduction and classification technique that:
1. Finds linear combinations of features that best separate classes
2. Maximizes between-class variance while minimizing within-class variance
3. Useful when data is normally distributed

Mathematical Concept:
- Between-class scatter matrix: S_B = Σ (μ_i - μ) * (μ_i - μ)^T
- Within-class scatter matrix: S_W = Σ (x - μ_i) * (x - μ_i)^T
- Objective: Maximize S_B / S_W
- Solution: Eigendecomposition of S_W^-1 * S_B
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis for dimensionality reduction and classification.

    Parameters:
    -----------
    n_components : int, default=2
        Number of linear discriminants (components)
    """

    def __init__(self, n_components=2):
        """Initialize LDA parameters."""
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
        self.class_mean = None
        self.overall_mean = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit LDA model by computing discriminant components.

        Algorithm:
        Step 1: Calculate overall mean and class means
        Step 2: Compute between-class scatter matrix (S_B)
        Step 3: Compute within-class scatter matrix (S_W)
        Step 4: Solve eigenvalue problem: S_W^-1 * S_B * w = λ * w
        Step 5: Select top n_components eigenvectors

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels

        Returns:
        --------
        self : fitted model
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Step 1: Calculate means
        self.overall_mean = np.mean(X, axis=0)
        self.class_mean = {}

        for cls in self.classes:
            self.class_mean[cls] = np.mean(X[y == cls], axis=0)

        # Step 2: Compute between-class scatter matrix (S_B)
        S_B = np.zeros((n_features, n_features))

        for cls in self.classes:
            n_cls = np.sum(y == cls)
            mean_diff = self.class_mean[cls] - self.overall_mean
            S_B += n_cls * np.outer(mean_diff, mean_diff)

        # Step 3: Compute within-class scatter matrix (S_W)
        S_W = np.zeros((n_features, n_features))

        for cls in self.classes:
            X_cls = X[y == cls]
            for x in X_cls:
                mean_diff = x - self.class_mean[cls]
                S_W += np.outer(mean_diff, mean_diff)

        # Step 4: Solve eigenvalue problem
        # S_W might be singular, use pseudo-inverse
        S_W_inv = np.linalg.pinv(S_W)
        eigen_matrix = S_W_inv @ S_B

        eigenvalues, eigenvectors = np.linalg.eig(eigen_matrix)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Select top n_components
        # Limit to min(n_components, n_classes-1)
        n_comp = min(self.n_components, n_classes - 1)
        self.components = eigenvectors[:, :n_comp]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:n_comp] / (total_variance + 1e-10)

        print(f"LDA fitted with {n_comp} components")
        print(f"Classes: {self.classes}")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")

        return self

    def transform(self, X):
        """
        Project data onto LDA components.

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

        return X @ self.components

    def fit_transform(self, X, y):
        """Fit model and transform data."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Classify samples using LDA.

        Find the class whose mean is closest to the sample in LDA space.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        X_transformed = self.transform(X)
        predictions = []

        # Transform class means to LDA space
        class_means_transformed = {}
        for cls in self.classes:
            class_means_transformed[cls] = self.class_mean[cls] @ self.components

        # Classify based on nearest class mean
        for x_transformed in X_transformed:
            distances = {}
            for cls in self.classes:
                dist = np.linalg.norm(x_transformed - class_means_transformed[cls])
                distances[cls] = dist

            prediction = min(distances, key=distances.get)
            predictions.append(prediction)

        return np.array(predictions)


# ============================================================================
# SAMPLE USAGE: LDA for Dimensionality Reduction and Classification
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LINEAR DISCRIMINANT ANALYSIS (LDA) - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load iris dataset
    print("\n[Step 1] Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Standardize features
    print("\n[Step 3] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Fit LDA
    print("\n[Step 4] Fitting LDA with 2 components...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # Step 5: Classification
    print("\n[Step 5] Classification using LDA...")
    y_pred_train = lda.predict(X_train_scaled)
    y_pred_test = lda.predict(X_test_scaled)

    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy:  {test_accuracy:.4f}")

    # Step 6: Component analysis
    print("\n[Step 6] LDA Component Analysis:")
    print(f"Explained variance ratio: {lda.explained_variance_ratio}")
    print(f"Cumulative variance: {np.sum(lda.explained_variance_ratio):.4f}")

    # Step 7: Visualizations
    print("\n[Step 7] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: LDA projection
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y))))
    for cls in np.unique(y):
        mask = y_train == cls
        axes[0, 0].scatter(
            X_train_lda[mask, 0],
            X_train_lda[mask, 1],
            c=[colors[cls]],
            label=iris.target_names[cls],
            s=100,
            alpha=0.7,
            edgecolors="k",
        )

    axes[0, 0].set_xlabel("LD1", fontsize=11)
    axes[0, 0].set_ylabel("LD2", fontsize=11)
    axes[0, 0].set_title(
        "LDA Projection (Training Data)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Test set projection
    for cls in np.unique(y):
        mask = y_test == cls
        axes[0, 1].scatter(
            X_test_lda[mask, 0],
            X_test_lda[mask, 1],
            c=[colors[cls]],
            label=iris.target_names[cls],
            s=100,
            alpha=0.7,
            edgecolors="k",
        )

    axes[0, 1].set_xlabel("LD1", fontsize=11)
    axes[0, 1].set_ylabel("LD2", fontsize=11)
    axes[0, 1].set_title("LDA Projection (Test Data)", fontsize=12, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Explained variance
    axes[1, 0].bar(
        range(len(lda.explained_variance_ratio)),
        lda.explained_variance_ratio,
        color="steelblue",
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 0].set_xlabel("Linear Discriminant", fontsize=11)
    axes[1, 0].set_ylabel("Explained Variance Ratio", fontsize=11)
    axes[1, 0].set_title(
        "Explained Variance by Component", fontsize=12, fontweight="bold"
    )
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Accuracy comparison
    accuracies = [train_accuracy, test_accuracy]
    datasets = ["Training", "Testing"]
    bars = axes[1, 1].bar(
        datasets,
        accuracies,
        color=["skyblue", "orange"],
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 1].set_ylabel("Accuracy", fontsize=11)
    axes[1, 1].set_title("Classification Accuracy", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("lda_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'lda_visualization.png'")
    plt.show()

    # Step 8: Confusion matrix
    print("\n[Step 8] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    print("\n" + "=" * 70)
    print("LDA completed successfully!")
    print("=" * 70)
