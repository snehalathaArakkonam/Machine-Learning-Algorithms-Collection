"""
Gradient Boosting Classifier Implementation
===========================================
Gradient Boosting sequentially builds an ensemble of decision trees,
where each new tree corrects residual errors made by previous trees.

Algorithm:
1. Train initial tree on actual labels
2. For each iteration:
    - Calculate residuals (actual - predictions)
    - Train new tree to predict residuals
    - Update predictions by adding scaled residuals
    - Use learning rate to prevent overfitting
3. Final prediction: sum of all tree predictions

Key Feature: Uses gradient descent to minimize loss function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier Implementation.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Shrinks contribution of each tree
    max_depth : int, default=3
        Maximum depth of trees
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Initialize Gradient Boosting parameters."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None

    def fit(self, X, y):
        """
        Train Gradient Boosting Classifier.

        Algorithm:
        Step 1: Initialize with log-odds for binary classification
        Step 2: For each boosting iteration:
            a. Calculate residuals using logistic loss
            b. Fit tree to residuals
            c. Update predictions

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Binary labels (0 or 1)
        """
        n_samples = X.shape[0]

        # Step 1: Initialize predictions
        pos_class_ratio = np.mean(y)
        self.init_prediction = np.log(pos_class_ratio / (1 - pos_class_ratio))

        # Current predictions (in log-odds space)
        current_predictions = np.full(n_samples, self.init_prediction, dtype=float)

        print(f"Training Gradient Boosting with {self.n_estimators} trees...")

        # Step 2: Boosting iterations
        for iteration in range(self.n_estimators):
            # Calculate probabilities from log-odds
            proba = 1 / (1 + np.exp(-current_predictions))

            # Calculate residuals (negative gradient of log loss)
            residuals = y - proba

            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)

            # Get tree predictions
            tree_pred = tree.predict(X)

            # Update current predictions with learning rate
            current_predictions += self.learning_rate * tree_pred

            # Store tree
            self.trees.append(tree)

            if (iteration + 1) % 20 == 0:
                proba_iter = 1 / (1 + np.exp(-current_predictions))
                y_pred_iter = (proba_iter >= 0.5).astype(int)
                acc = accuracy_score(y, y_pred_iter)
                print(f"  Iteration {iteration + 1}: Accuracy = {acc:.4f}")

        print(f"✓ Training complete!")

        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        # Get probability predictions
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Returns:
        --------
        probabilities : array, shape (n_samples, 2)
            Probabilities for each class
        """
        predictions = np.full(X.shape[0], self.init_prediction, dtype=float)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        # Convert log-odds to probabilities
        proba_class1 = 1 / (1 + np.exp(-predictions))
        proba_class0 = 1 - proba_class1

        return np.column_stack([proba_class0, proba_class1])


# ============================================================================
# SAMPLE USAGE: Gradient Boosting Classifier
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GRADIENT BOOSTING CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load breast cancer dataset
    print("\n[Step 1] Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")

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

    # Step 4: Train Gradient Boosting
    print("\n[Step 4] Training Gradient Boosting Classifier...")
    print("Parameters: n_estimators=100, learning_rate=0.1, max_depth=3")

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train_scaled, y_train)

    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Step 6: Calculate metrics
    print("\n[Step 6] Calculating metrics...")
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {train_acc:.4f} ({int(train_acc*100)}%)")
    print(f"Testing Accuracy:  {test_acc:.4f} ({int(test_acc*100)}%)")

    # Step 7: Confusion matrix
    print("\n[Step 7] Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"               Predicted 0  Predicted 1")
    print(f"Actual 0:      {cm[0, 0]:>11}  {cm[0, 1]:>11}")
    print(f"Actual 1:      {cm[1, 0]:>11}  {cm[1, 1]:>11}")

    # Step 8: Additional metrics
    print("\n[Step 8] Additional Metrics:")
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy comparison
    accuracies = [train_acc, test_acc]
    datasets = ["Training", "Testing"]
    bars = axes[0, 0].bar(
        datasets,
        accuracies,
        color=["skyblue", "orange"],
        edgecolor="black",
        linewidth=2,
    )
    axes[0, 0].set_ylabel("Accuracy", fontsize=11)
    axes[0, 0].set_title(
        "Gradient Boosting: Train vs Test", fontsize=12, fontweight="bold"
    )
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

    plt.colorbar(im, ax=axes[0, 1])

    # Plot 3: Probability distribution
    axes[1, 0].hist(
        y_proba[y_test == 0, 1],
        bins=20,
        alpha=0.6,
        label="Class 0",
        color="blue",
        edgecolor="black",
    )
    axes[1, 0].hist(
        y_proba[y_test == 1, 1],
        bins=20,
        alpha=0.6,
        label="Class 1",
        color="red",
        edgecolor="black",
    )
    axes[1, 0].axvline(
        0.5, color="black", linestyle="--", linewidth=2, label="Decision boundary"
    )
    axes[1, 0].set_xlabel("Predicted Probability (Class 1)", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title(
        "Predicted Probability Distribution", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Metrics comparison
    metrics_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
    metrics_values = [precision, recall, f1, test_acc]
    bars = axes[1, 1].bar(
        metrics_names, metrics_values, color="steelblue", edgecolor="black", linewidth=2
    )
    axes[1, 1].set_ylabel("Score", fontsize=11)
    axes[1, 1].set_title("Performance Metrics", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("gradient_boosting_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'gradient_boosting_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Gradient Boosting Classifier training completed successfully!")
    print("=" * 70)
