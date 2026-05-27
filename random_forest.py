"""
Random Forest Classifier Implementation
========================================
Random Forest is an ensemble learning method that builds multiple decision trees
and combines their predictions. It reduces overfitting through:
1. Bootstrap aggregating (Bagging): Each tree trained on random sample with replacement
2. Random feature selection: Each split considers random subset of features
3. Majority voting: Final prediction is most common prediction among all trees

Why It Works:
- Multiple diverse trees reduce variance and overfitting
- Each tree is deep and may overfit, but averaging reduces this effect
- Captures complex non-linear relationships better than single tree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter


class Node:
    """Represents a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """Simple Decision Tree for use in Random Forest."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        # Stopping criteria
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._most_common_label(y))

        if n_samples < self.min_samples_split or n_classes == 1:
            return Node(value=self._most_common_label(y))

        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_entropy = self._entropy(y)

        for feature in range(n_features):
            X_col = X[:, feature]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                left_mask = X_col <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                entropy_left = self._entropy(y[left_mask])
                entropy_right = self._entropy(y[right_mask])

                weighted_entropy = (n_left / n_samples) * entropy_left + (
                    n_right / n_samples
                ) * entropy_right

                information_gain = parent_entropy - weighted_entropy

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return Node(value=self._most_common_label(y))

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
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForestClassifier:
    """
    Random Forest Classifier Implementation.

    Algorithm Steps:
    1. For each tree (n_estimators):
        a. Create bootstrap sample (random sample with replacement)
        b. Randomly select subset of features for each split
        c. Build decision tree without pruning
    2. For prediction: Get prediction from each tree, return majority vote

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=None
        Maximum depth of trees
    min_samples_split : int, default=2
        Minimum samples required to split
    max_features : str or int, default='sqrt'
        Number of features to consider at each split
        ('sqrt': sqrt(n_features), 'log2': log2(n_features), int: specific number)
    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=None,
    ):
        """Initialize Random Forest parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Build Random Forest by training multiple decision trees.

        Step 1: For each tree to build:
            - Create bootstrap sample (sampling with replacement)
            - Train decision tree on this sample
        Step 2: Calculate feature importances based on information gain

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        """
        n_samples, n_features = X.shape

        # Determine number of features to consider at each split
        if self.max_features == "sqrt":
            n_features_split = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_features_split = int(np.log2(n_features))
        else:
            n_features_split = self.max_features

        # Step 1: Build multiple decision trees
        print(f"Building {self.n_estimators} trees...")
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} trees")

            # Bootstrap sample (sample with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        print(f"✓ Training completed!")

        # Step 2: Calculate feature importances
        # Initialize importances
        self.feature_importances = np.zeros(n_features)

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting.

        Mathematical Approach:
        1. Get prediction from each tree
        2. Collect all predictions (voting)
        3. Return class with highest vote count

        For each sample:
            predictions = [tree.predict(sample) for each tree]
            final_prediction = mode(predictions)  # Most common

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if not self.trees:
            raise ValueError("Model must be fit before predictions.")

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority voting: take most common prediction for each sample
        predictions = []
        for i in range(X.shape[0]):
            # Get all predictions for this sample
            sample_predictions = tree_predictions[:, i]
            # Find most common prediction
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Returns probability as proportion of trees voting for each class.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        classes = np.unique(tree_predictions)
        probabilities = []

        for i in range(X.shape[0]):
            sample_predictions = tree_predictions[:, i]
            proba = []
            for cls in classes:
                prob = np.mean(sample_predictions == cls)
                proba.append(prob)
            probabilities.append(proba)

        return np.array(probabilities)


# ============================================================================
# SAMPLE USAGE: Random Forest on Iris Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM FOREST CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load iris dataset
    print("\n[Step 1] Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {iris.feature_names}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Create and train Random Forest
    print("\n[Step 3] Training Random Forest Classifier...")
    print("Parameters: n_estimators=100, max_depth=None, max_features='sqrt'")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features="sqrt", random_state=42
    )
    model.fit(X_train, y_train)

    # Step 4: Make predictions
    print("\n[Step 4] Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba = model.predict_proba(X_test[:5])

    # Step 5: Calculate accuracy
    print("\n[Step 5] Calculating accuracy metrics...")
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({int(train_accuracy*100)}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({int(test_accuracy*100)}%)")

    # Step 6: Classification report
    print("\n[Step 6] Per-class Performance:")
    print(f"{'Class':<15} {'Accuracy':<15}")
    print("-" * 30)

    for i in range(len(np.unique(y))):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred_test[mask] == y_test[mask])
            print(f"{iris.target_names[i]:<15} {class_accuracy:<15.4f}")

    # Step 7: Sample predictions with probabilities
    print(
        "\n[Step 7] Sample Predictions with Class Probabilities (First 5 test samples):"
    )
    print(f"{'Actual':<12} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 36)

    for i in range(min(5, len(y_test))):
        max_proba = np.max(y_proba[i])
        actual = iris.target_names[y_test[i]]
        predicted = iris.target_names[y_pred_test[i]]
        print(f"{actual:<12} {predicted:<12} {max_proba:<12.2%}")

    # Step 8: Confusion matrix
    print("\n[Step 8] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy comparison
    accuracies = [train_accuracy, test_accuracy]
    datasets = ["Training", "Testing"]
    bars = axes[0, 0].bar(
        datasets,
        accuracies,
        color=["skyblue", "orange"],
        edgecolor="black",
        linewidth=2,
    )
    axes[0, 0].set_ylabel("Accuracy", fontsize=11)
    axes[0, 0].set_title("Training vs Testing Accuracy", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 2: Confusion matrix heatmap
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
    axes[1, 0].set_title(
        "Per-Class Accuracy (Test Set)", fontsize=12, fontweight="bold"
    )
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

    # Plot 4: Prediction distribution
    pred_counts = np.bincount(y_pred_test)
    true_counts = np.bincount(y_test)

    x = np.arange(len(np.unique(y)))
    width = 0.35

    bars1 = axes[1, 1].bar(
        x - width / 2,
        true_counts,
        width,
        label="True",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[1, 1].bar(
        x + width / 2,
        pred_counts,
        width,
        label="Predicted",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[1, 1].set_xlabel("Class", fontsize=11)
    axes[1, 1].set_ylabel("Count", fontsize=11)
    axes[1, 1].set_title(
        "True vs Predicted Distribution (Test Set)", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"Class {i}" for i in range(len(np.unique(y)))])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("random_forest_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'random_forest_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Random Forest Classifier training completed successfully!")
    print("=" * 70)
