"""
K-Nearest Neighbors (KNN) Classifier Implementation
===================================================
KNN is a simple instance-based (lazy) learning algorithm that classifies samples
based on their k nearest neighbors in the training set. A sample is classified
based on the majority class of its k neighbors.

Key Concepts:
1. Distance metric: Euclidean, Manhattan, or other distance measures
2. k parameter: Number of neighbors to consider
3. Lazy learning: No training phase, all computation during prediction
4. Decision boundary: Non-linear, adapts to local data distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


class KNearestNeighbors:
    """
    K-Nearest Neighbors Classifier Implementation.

    Parameters:
    -----------
    k : int, default=3
        Number of neighbors to use for classification
    distance_metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'minkowski')
    weights : str, default='uniform'
        Weight function for predictions
        - 'uniform': all neighbors weighted equally
        - 'distance': neighbors weighted by 1/distance
    """

    def __init__(self, k=3, distance_metric="euclidean", weights="uniform"):
        """Initialize KNN parameters."""
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data (lazy learning - no actual training).

        Step 1: Simply store training features and labels
        Step 2: No parameters to optimize

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        """
        # Step 1: Store training data
        self.X_train = X
        self.y_train = y

        print(f"KNN initialized with k={self.k}")
        print(f"Training samples stored: {X.shape[0]}")

        return self

    def predict(self, X):
        """
        Predict class labels for samples.

        Algorithm:
        Step 1: For each sample, compute distances to all training samples
        Step 2: Find k nearest neighbors (smallest distances)
        Step 3: Use majority voting to determine class

        Distance Metrics:
        - Euclidean: sqrt(sum((x_i - y_i)^2))
        - Manhattan: sum(|x_i - y_i|)
        - Minkowski: (sum(|x_i - y_i|^p))^(1/p)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if self.X_train is None:
            raise ValueError("Model must be fit before predictions.")

        predictions = []

        # Step 1-3: For each test sample
        for sample in X:
            # Compute distances to all training samples
            distances = self._compute_distances(sample, self.X_train)

            # Find k nearest neighbors
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]

            # Majority voting
            if self.weights == "uniform":
                # Equal weight for all neighbors
                prediction = Counter(k_labels).most_common(1)[0][0]
            else:  # weights == 'distance'
                # Weight by 1/distance (closer neighbors have more influence)
                weights_list = 1.0 / (k_distances + 1e-10)
                weighted_votes = {}
                for label, weight in zip(k_labels, weights_list):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                prediction = max(weighted_votes, key=weighted_votes.get)

            predictions.append(prediction)

        return np.array(predictions)

    def _compute_distances(self, sample, X):
        """
        Compute distances between a sample and all training samples.

        Parameters:
        -----------
        sample : array, shape (n_features,)
            Single sample
        X : array, shape (n_samples, n_features)
            Training samples

        Returns:
        --------
        distances : array, shape (n_samples,)
            Distances from sample to each training sample
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance
            return np.sqrt(np.sum((X - sample) ** 2, axis=1))

        elif self.distance_metric == "manhattan":
            # Manhattan distance
            return np.sum(np.abs(X - sample), axis=1)

        elif self.distance_metric == "minkowski":
            # Minkowski distance (generalization, p=2 equals Euclidean)
            p = 2
            return np.sum(np.abs(X - sample) ** p, axis=1) ** (1 / p)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict_proba(self, X):
        """
        Return probability estimates for each class.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Probability for each class
        """
        if self.X_train is None:
            raise ValueError("Model must be fit before predictions.")

        classes = np.unique(self.y_train)
        probabilities = []

        for sample in X:
            distances = self._compute_distances(sample, self.X_train)
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_indices]

            # Count occurrences of each class
            counts = Counter(k_labels)
            proba = []
            for cls in classes:
                proba.append(counts.get(cls, 0) / self.k)
            probabilities.append(proba)

        return np.array(probabilities)


# ============================================================================
# SAMPLE USAGE: KNN Classifier on Iris Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("K-NEAREST NEIGHBORS (KNN) CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load iris dataset
    print("\n[Step 1] Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Standardize features (important for distance-based algorithms)
    print("\n[Step 3] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Test different k values
    print("\n[Step 4] Testing different k values to find optimal...")
    k_values = [1, 3, 5, 7, 9, 15]
    accuracies = []

    for k in k_values:
        model = KNearestNeighbors(k=k, distance_metric="euclidean", weights="uniform")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        print(f"  k={k:2d}: Accuracy = {accuracy:.4f}")

    # Step 5: Use optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    print(f"\n[Step 5] Using optimal k={optimal_k} for final model...")

    model = KNearestNeighbors(
        k=optimal_k, distance_metric="euclidean", weights="uniform"
    )
    model.fit(X_train_scaled, y_train)

    # Step 6: Make predictions
    print("\n[Step 6] Making predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled[:10])

    # Step 7: Calculate accuracy
    print("\n[Step 7] Calculating accuracy metrics...")
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({int(train_accuracy*100)}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({int(test_accuracy*100)}%)")

    # Step 8: Per-class accuracy
    print("\n[Step 8] Per-class Accuracy (Test Set):")
    print(f"{'Class':<15} {'Accuracy':<15}")
    print("-" * 30)

    for i in range(len(np.unique(y))):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred_test[mask] == y_test[mask])
            print(f"{iris.target_names[i]:<15} {class_accuracy:<15.4f}")

    # Step 9: Sample predictions with probabilities
    print(
        "\n[Step 9] Sample Predictions with Class Probabilities (First 10 test samples):"
    )
    print(f"{'Actual':<15} {'Predicted':<15} {'Confidence':<12}")
    print("-" * 42)

    for i in range(min(10, len(y_test))):
        max_proba = np.max(y_proba[i])
        actual = iris.target_names[y_test[i]]
        predicted = iris.target_names[y_pred_test[i]]
        is_correct = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(f"{actual:<15} {predicted:<15} {max_proba:<12.2%} {is_correct}")

    # Step 10: Confusion matrix
    print("\n[Step 10] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    # Step 11: Visualization
    print("\n[Step 11] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: K value vs Accuracy
    axes[0, 0].plot(
        k_values, accuracies, marker="o", linewidth=2, markersize=8, color="steelblue"
    )
    axes[0, 0].scatter(
        [optimal_k],
        [max(accuracies)],
        color="red",
        s=200,
        marker="*",
        label=f"Optimal k={optimal_k}",
        zorder=5,
    )
    axes[0, 0].set_xlabel("k Value", fontsize=11)
    axes[0, 0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0, 0].set_title("Accuracy vs k Parameter", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xticks(k_values)

    # Plot 2: Confusion matrix
    im = axes[0, 1].imshow(cm, cmap="Blues", aspect="auto")
    axes[0, 1].set_xlabel("Predicted Label", fontsize=11)
    axes[0, 1].set_ylabel("True Label", fontsize=11)
    axes[0, 1].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(range(len(np.unique(y))))
    axes[0, 1].set_yticks(range(len(np.unique(y))))

    for i in range(len(np.unique(y))):
        for j in range(len(np.unique(y))):
            text = axes[0, 1].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    # Plot 3: Per-class accuracy
    class_accuracies = []
    class_names = []
    for i in range(len(np.unique(y))):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.mean(y_pred_test[mask] == y_test[mask])
            class_accuracies.append(acc)
            class_names.append(iris.target_names[i])

    bars = axes[1, 0].barh(
        class_names,
        class_accuracies,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 0].set_xlabel("Accuracy", fontsize=11)
    axes[1, 0].set_title("Per-Class Accuracy", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlim([0, 1])

    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 0].text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.2%}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 4: Training vs Testing accuracy
    accuracies_compare = [train_accuracy, test_accuracy]
    datasets = ["Training", "Testing"]
    colors = ["skyblue", "orange"]
    bars = axes[1, 1].bar(
        datasets, accuracies_compare, color=colors, edgecolor="black", linewidth=2
    )
    axes[1, 1].set_ylabel("Accuracy", fontsize=11)
    axes[1, 1].set_title(
        f"KNN (k={optimal_k}): Train vs Test Accuracy", fontsize=12, fontweight="bold"
    )
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
    plt.savefig("knn_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'knn_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("K-Nearest Neighbors training completed successfully!")
    print("=" * 70)
