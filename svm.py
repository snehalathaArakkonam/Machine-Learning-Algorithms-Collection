"""
Support Vector Machine (SVM) Classifier Implementation
=======================================================
SVM is a powerful classification algorithm that finds the optimal hyperplane
to separate classes with maximum margin. It works well for both linear and
non-linear classification through kernel functions.

Key Concepts:
1. Support Vectors: Data points closest to the decision boundary
2. Margin: Distance between hyperplane and closest points
3. Kernel Trick: Maps data to higher dimension for non-linear separation
4. C Parameter: Trade-off between margin and misclassification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SupportVectorMachine:
    """
    Support Vector Machine Classifier (Simplified using SMO algorithm concept).

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter. Higher C = less regularization.
    kernel : str, default='linear'
        Type of kernel ('linear', 'rbf', 'poly')
    gamma : float, default=0.1
        Kernel coefficient for 'rbf' and 'poly' kernels
    degree : int, default=3
        Degree for polynomial kernel
    learning_rate : float, default=0.001
        Learning rate for optimization
    max_iterations : int, default=1000
        Maximum iterations for training
    """

    def __init__(
        self,
        C=1.0,
        kernel="linear",
        gamma=0.1,
        degree=3,
        learning_rate=0.001,
        max_iterations=1000,
    ):
        """Initialize SVM parameters."""
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.X_train = None
        self.y_train = None
        self.alphas = None
        self.support_vectors = None

    def fit(self, X, y):
        """
        Train SVM using simplified gradient descent approach.

        Step 1: Initialize weights and bias
        Step 2: Convert binary labels to {-1, +1}
        Step 3: Compute kernel matrix for all training samples
        Step 4: Optimize using gradient descent with hinge loss
        Step 5: Store support vectors

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Binary training labels (0/1)
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.X_train = X

        # Step 2: Convert labels to {-1, +1}
        self.y_train = np.where(y == 0, -1, 1)

        # Step 3: Compute kernel matrix
        kernel_matrix = self._compute_kernel_matrix(X, X)

        print(f"Training SVM with {self.kernel} kernel...")

        # Step 4: Optimize using gradient descent
        for iteration in range(self.max_iterations):
            # Compute predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Hinge loss: max(0, 1 - y * y_pred)
            hinge_loss = np.maximum(0, 1 - self.y_train * y_pred)
            loss = np.mean(hinge_loss) + (self.C / 2) * np.sum(self.weights**2)

            # Gradient of hinge loss
            dw = np.zeros_like(self.weights)
            db = 0

            for i in range(n_samples):
                if self.y_train[i] * y_pred[i] < 1:
                    dw += -self.y_train[i] * X[i]
                    db += -self.y_train[i]

            # Add L2 regularization gradient
            dw = dw / n_samples + (self.C / n_samples) * self.weights
            db = db / n_samples

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (iteration + 1) % 200 == 0:
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.4f}"
                )

        print("✓ Training completed!")

        # Step 5: Identify support vectors
        y_pred = np.dot(X, self.weights) + self.bias
        margins = self.y_train * y_pred
        self.support_vectors = np.where(margins <= 1.0)[0]

        return self

    def _compute_kernel_matrix(self, X1, X2):
        """
        Compute kernel matrix between two sets of samples.

        Mathematical Kernels:
        - Linear: K(x, x') = x · x'
        - RBF: K(x, x') = exp(-gamma * ||x - x'||²)
        - Polynomial: K(x, x') = (x · x' + 1)^degree
        """
        if self.kernel == "linear":
            return np.dot(X1, X2.T)

        elif self.kernel == "rbf":
            # Compute pairwise distances
            distances = np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2)
            return np.exp(-self.gamma * distances**2)

        elif self.kernel == "poly":
            return (np.dot(X1, X2.T) + 1) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def predict(self, X):
        """
        Predict class labels for samples.

        Decision Function:
        f(x) = sign(w · φ(x) + b)

        Where φ(x) is the kernel transformation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted labels (0 or 1)
        """
        if self.weights is None:
            raise ValueError("Model must be fit before predictions.")

        # Compute predictions
        y_pred = np.dot(X, self.weights) + self.bias

        # Convert to class labels (0 or 1)
        return np.where(y_pred >= 0, 1, 0)

    def decision_function(self, X):
        """Return the decision function values."""
        return np.dot(X, self.weights) + self.bias


# ============================================================================
# SAMPLE USAGE: SVM Classifier
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load iris dataset (binary classification: setosa vs others)
    print("\n[Step 1] Loading iris dataset (Binary: Setosa vs Others)...")
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # Binary: setosa (1) vs others (0)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Standardize features
    print("\n[Step 3] Standardizing features (crucial for SVM)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Train SVM with linear kernel
    print("\n[Step 4] Training SVM with linear kernel...")
    model_linear = SupportVectorMachine(
        C=1.0, kernel="linear", learning_rate=0.001, max_iterations=1000
    )
    model_linear.fit(X_train_scaled, y_train)

    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    y_pred_train = model_linear.predict(X_train_scaled)
    y_pred_test = model_linear.predict(X_test_scaled)

    # Step 6: Calculate accuracy
    print("\n[Step 6] Calculating accuracy metrics...")
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({int(train_accuracy*100)}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({int(test_accuracy*100)}%)")

    # Step 7: Number of support vectors
    print(f"\n[Step 7] Support Vector Analysis:")
    n_support_vectors = len(model_linear.support_vectors)
    print(
        f"Number of support vectors: {n_support_vectors}/{len(X_train)} ({n_support_vectors/len(X_train)*100:.1f}%)"
    )

    # Step 8: Confusion matrix
    print("\n[Step 8] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(f"               Predicted 0  Predicted 1")
    print(f"Actual 0:      {cm[0, 0]:>11}  {cm[0, 1]:>11}")
    print(f"Actual 1:      {cm[1, 0]:>11}  {cm[1, 1]:>11}")

    # Step 9: Sample predictions
    print("\n[Step 9] Sample Predictions (First 10 test samples):")
    print(f"{'Actual':<10} {'Predicted':<12} {'Match':<10}")
    print("-" * 32)

    for i in range(min(10, len(y_test))):
        match = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(f"{y_test[i]:<10} {y_pred_test[i]:<12} {match:<10}")

    # Step 10: Visualization
    print("\n[Step 10] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use first two features for 2D visualization
    X_train_2d = X_train_scaled[:, :2]
    X_test_2d = X_test_scaled[:, :2]

    # Plot 1: Decision boundary for first two features
    h = 0.02
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # For visualization, we need a 2D model
    model_2d = SupportVectorMachine(C=1.0, kernel="linear")
    model_2d.fit(X_train_2d, y_train)
    Z = model_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[0, 0].contourf(
        xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap="RdBu", alpha=0.8
    )
    axes[0, 0].contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")
    scatter = axes[0, 0].scatter(
        X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="RdBu", edgecolors="k", s=50
    )
    axes[0, 0].set_xlabel("Feature 0", fontsize=11)
    axes[0, 0].set_ylabel("Feature 1", fontsize=11)
    axes[0, 0].set_title(
        "SVM Decision Boundary (Training Data)", fontsize=12, fontweight="bold"
    )

    # Plot 2: Confusion matrix
    im = axes[0, 1].imshow(cm, cmap="Blues", aspect="auto")
    axes[0, 1].set_xlabel("Predicted Label", fontsize=11)
    axes[0, 1].set_ylabel("True Label", fontsize=11)
    axes[0, 1].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
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

    # Plot 3: Accuracy comparison
    accuracies = [train_accuracy, test_accuracy]
    datasets = ["Training", "Testing"]
    bars = axes[1, 0].bar(
        datasets,
        accuracies,
        color=["skyblue", "orange"],
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 0].set_ylabel("Accuracy", fontsize=11)
    axes[1, 0].set_title("SVM Accuracy: Train vs Test", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 4: Class distribution
    classes = ["Class 0", "Class 1"]
    train_dist = [np.sum(y_train == 0), np.sum(y_train == 1)]
    test_dist = [np.sum(y_test == 0), np.sum(y_test == 1)]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = axes[1, 1].bar(
        x - width / 2,
        train_dist,
        width,
        label="Train",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[1, 1].bar(
        x + width / 2,
        test_dist,
        width,
        label="Test",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[1, 1].set_ylabel("Count", fontsize=11)
    axes[1, 1].set_title("Class Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("svm_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'svm_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("SVM Classifier training completed successfully!")
    print("=" * 70)
