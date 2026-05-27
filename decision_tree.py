"""
Decision Tree Classifier Implementation
========================================
A decision tree is a supervised learning algorithm that builds a tree-like model
of decisions based on feature values. It splits data recursively to create regions
where each region is assigned a class label.

How It Works:
1. Start with all samples at root node
2. Find best feature and threshold that maximizes information gain
3. Split samples into two groups (left < threshold, right >= threshold)
4. Recursively repeat for each group until stopping criterion is met
5. Assign class labels to leaf nodes based on majority vote
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class Node:
    """Represents a node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Parameters:
        -----------
        feature : int
            Index of feature to split on (None for leaf nodes)
        threshold : float
            Threshold value for the split
        left : Node
            Left child node
        right : Node
            Right child node
        value : int
            Class label for leaf nodes
        """
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Class label (for leaf nodes)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier Implementation.

    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree. None = no limit.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """Initialize Decision Tree parameters."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Build decision tree classifier.

        Step 1: Initialize tree building process
        Step 2: Recursively partition data using information gain
        Step 3: Stop when max_depth or min_samples_split is reached
        Step 4: Assign class labels to leaf nodes

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        """
        # Step 1: Start building tree from root
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Mathematical Concept - Information Gain:
        IG = Entropy(parent) - Weighted_Avg(Entropy(children))

        Entropy = -Σ(p_i * log2(p_i)) where p_i is proportion of class i

        We choose the split that maximizes information gain.
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        # Step 2: Check stopping criteria
        # Criterion 1: Reach maximum depth
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Criterion 2: Minimum samples to split
        if n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Criterion 3: All samples have the same class
        if n_classes == 1:
            leaf_value = y[0]
            return Node(value=leaf_value)

        # Step 3: Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_entropy = self._entropy(y)

        # Try each feature
        for feature in range(n_features):
            X_col = X[:, feature]
            thresholds = np.unique(X_col)

            # Try each threshold
            for threshold in thresholds:
                # Split samples
                left_mask = X_col <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                entropy_left = self._entropy(y[left_mask])
                entropy_right = self._entropy(y[right_mask])

                weighted_entropy = (n_left / n_samples) * entropy_left + (
                    n_right / n_samples
                ) * entropy_right

                information_gain = parent_entropy - weighted_entropy

                # Update best split if this is better
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        # If no good split found, create leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Step 4: Split data and recursively build subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _entropy(self, y):
        """
        Calculate entropy for a group of samples.

        Entropy measures disorder/impurity:
        - Entropy = 0: Pure node (only one class)
        - Entropy = 1: Completely mixed classes (binary classification)
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _most_common_label(self, y):
        """Return the most common class label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X.

        For each sample, traverse the tree from root to leaf:
        - At each node, go left if X[feature] <= threshold, else go right
        - Return the class label of the reached leaf node

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if self.tree is None:
            raise ValueError("Model must be fit before predictions.")

        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse tree to get prediction for a single sample."""
        # Base case: leaf node
        if node.value is not None:
            return node.value

        # Recursive case: internal node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


# ============================================================================
# SAMPLE USAGE: Decision Tree on Iris Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DECISION TREE CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load the iris dataset
    print("\n[Step 1] Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class names: {iris.target_names}")

    # Step 2: Split data into training and testing sets
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Step 3: Create and train Decision Tree
    print("\n[Step 3] Training Decision Tree Classifier (max_depth=5)...")
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)

    # Step 4: Make predictions
    print("\n[Step 4] Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Step 5: Calculate accuracy
    print("\n[Step 5] Calculating accuracy metrics...")
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({int(train_accuracy*100)}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({int(test_accuracy*100)}%)")

    # Step 6: Confusion matrix
    print("\n[Step 6] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    # Step 7: Classification report
    print("\n[Step 7] Classification Report:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 46)

    for i in range(len(np.unique(y))):
        true_positives = np.sum((y_pred_test == i) & (y_test == i))
        false_positives = np.sum((y_pred_test == i) & (y_test != i))
        false_negatives = np.sum((y_pred_test != i) & (y_test == i))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        print(
            f"{iris.target_names[i]:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}"
        )

    # Step 8: Sample predictions
    print("\n[Step 8] Sample Predictions (First 15 test samples):")
    print(f"{'Actual':<15} {'Predicted':<15} {'Correct':<10}")
    print("-" * 40)

    for i in range(min(15, len(y_test))):
        actual_name = iris.target_names[y_test[i]]
        predicted_name = iris.target_names[y_pred_test[i]]
        is_correct = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(f"{actual_name:<15} {predicted_name:<15} {is_correct:<10}")

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    # Focus on first two features for 2D visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training data with decision boundary
    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Note: This requires extending predict to handle partial features
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            # Create sample with mesh coordinates and average for other features
            sample = np.concatenate([[xx[i, j], yy[i, j]], X_train[:, 2:].mean(axis=0)])
            Z[i, j] = model.predict(sample.reshape(1, -1))[0]

    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = axes[0].scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="viridis",
        edgecolors="k",
        s=50,
    )
    axes[0].set_xlabel(iris.feature_names[0], fontsize=11)
    axes[0].set_ylabel(iris.feature_names[1], fontsize=11)
    axes[0].set_title(
        "Decision Tree Decision Boundary (Training Data)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(scatter, ax=axes[0], label="Class")

    # Plot 2: Confusion Matrix Heatmap
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    im = axes[1].imshow(cm, cmap="Blues", aspect="auto")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(len(np.unique(y))))
    axes[1].set_yticks(range(len(np.unique(y))))

    # Add text annotations
    for i in range(len(np.unique(y))):
        for j in range(len(np.unique(y))):
            text = axes[1].text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12
            )

    plt.colorbar(im, ax=axes[1], label="Count")

    plt.tight_layout()
    plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'decision_tree_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Decision Tree Classifier training completed successfully!")
    print("=" * 70)
