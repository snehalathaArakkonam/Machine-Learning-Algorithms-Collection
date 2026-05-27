"""
AdaBoost (Adaptive Boosting) Classifier Implementation
======================================================
AdaBoost is an ensemble learning method that combines weak learners (usually decision trees)
into a strong classifier by adaptively adjusting sample weights.

Algorithm:
1. Initialize equal weights for all samples
2. For each iteration:
    - Train weak learner on weighted dataset
    - Calculate error rate
    - Calculate learner weight (alpha) based on accuracy
    - Update sample weights: increase for misclassified, decrease for correct
3. Final prediction: weighted majority vote of all weak learners

Key Concept: Focuses on hard-to-classify samples by increasing their weight.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) Classifier Implementation.

    Parameters:
    -----------
    n_estimators : int, default=50
        Number of weak learners
    learning_rate : float, default=1.0
        Shrinks contribution of each weak learner
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        """Initialize AdaBoost parameters."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weak_learners = []
        self.learner_weights = []

    def fit(self, X, y):
        """
        Train AdaBoost classifier.

        Algorithm:
        Step 1: Initialize sample weights (uniform)
        Step 2: For each weak learner iteration:
            a. Train weak learner on weighted dataset
            b. Calculate weighted error rate
            c. Calculate learner weight (alpha)
            d. Update sample weights

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Binary labels (0 or 1)
        """
        n_samples = X.shape[0]

        # Convert to {-1, +1} labels if needed
        y_binary = np.where(y == 0, -1, 1)

        # Step 1: Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples

        print(f"Training AdaBoost with {self.n_estimators} weak learners...")

        # Step 2: Iterative weak learner training
        for m in range(self.n_estimators):
            # Train weak learner (decision stump - depth 1)
            weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
            weak_learner.fit(X, y, sample_weight=sample_weights)

            # Get predictions
            y_pred = weak_learner.predict(X)
            y_pred_binary = np.where(y_pred == 0, -1, 1)

            # Calculate weighted error
            error = np.sum(sample_weights * (y_pred_binary != y_binary))

            # Avoid division by zero
            if error == 0:
                error = 1e-10
            if error >= 0.5:
                error = 0.5 - 1e-10

            # Calculate learner weight (alpha)
            alpha = self.learning_rate * (0.5 * np.log((1 - error) / error))

            # Update sample weights
            sample_weights *= np.exp(-alpha * y_binary * y_pred_binary)
            sample_weights /= np.sum(sample_weights)

            # Store weak learner and its weight
            self.weak_learners.append(weak_learner)
            self.learner_weights.append(alpha)

            if (m + 1) % 10 == 0:
                print(f"  Trained {m + 1} weak learners, Error: {error:.4f}")

        print(f"✓ Training complete!")

        return self

    def predict(self, X):
        """
        Predict class labels using AdaBoost ensemble.

        Algorithm:
        Final prediction = sign(Σ alpha_m * y_m(x))

        Where:
        - alpha_m: weight of weak learner m
        - y_m(x): prediction of weak learner m

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        predictions = np.zeros(X.shape[0])

        for alpha, weak_learner in zip(self.learner_weights, self.weak_learners):
            y_pred = weak_learner.predict(X)
            y_pred_binary = np.where(y_pred == 0, -1, 1)
            predictions += alpha * y_pred_binary

        return np.where(predictions >= 0, 1, 0)

    def predict_proba(self, X):
        """Return probability estimates."""
        predictions = np.zeros(X.shape[0])

        for alpha, weak_learner in zip(self.learner_weights, self.weak_learners):
            y_pred = weak_learner.predict(X)
            y_pred_binary = np.where(y_pred == 0, -1, 1)
            predictions += alpha * y_pred_binary

        # Convert to probabilities
        proba = 1 / (1 + np.exp(-2 * predictions))
        return np.column_stack([1 - proba, proba])


# ============================================================================
# SAMPLE USAGE: AdaBoost Classifier
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ADABOOST CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create binary classification dataset
    print("\n[Step 1] Creating binary classification dataset...")
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )

    print(f"Dataset shape: {X.shape}")
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

    # Step 4: Train AdaBoost
    print("\n[Step 4] Training AdaBoost Classifier...")
    model = AdaBoost(n_estimators=50, learning_rate=1.0)
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

    # Step 8: Learner weights
    print("\n[Step 8] Weak Learner Weights (Top 10):")
    top_indices = np.argsort(model.learner_weights)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  Learner {idx+1}: {model.learner_weights[idx]:.4f}")

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Weak learner weights
    axes[0, 0].bar(
        range(len(model.learner_weights)),
        model.learner_weights,
        color="steelblue",
        edgecolor="black",
        linewidth=1,
    )
    axes[0, 0].set_xlabel("Weak Learner Index", fontsize=11)
    axes[0, 0].set_ylabel("Weight", fontsize=11)
    axes[0, 0].set_title(
        "AdaBoost Weak Learner Weights", fontsize=12, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Cumulative weight evolution
    cumsum_weights = np.cumsum(model.learner_weights)
    axes[0, 1].plot(
        range(1, len(cumsum_weights) + 1),
        cumsum_weights,
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 1].set_xlabel("Number of Weak Learners", fontsize=11)
    axes[0, 1].set_ylabel("Cumulative Weight", fontsize=11)
    axes[0, 1].set_title("Cumulative Weight Evolution", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Accuracy comparison
    accuracies = [train_acc, test_acc]
    datasets = ["Training", "Testing"]
    bars = axes[1, 0].bar(
        datasets,
        accuracies,
        color=["skyblue", "orange"],
        edgecolor="black",
        linewidth=2,
    )
    axes[1, 0].set_ylabel("Accuracy", fontsize=11)
    axes[1, 0].set_title("AdaBoost: Train vs Test", fontsize=12, fontweight="bold")
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

    # Plot 4: Confusion matrix heatmap
    im = axes[1, 1].imshow(cm, cmap="Blues", aspect="auto")
    axes[1, 1].set_xlabel("Predicted Label", fontsize=11)
    axes[1, 1].set_ylabel("True Label", fontsize=11)
    axes[1, 1].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("adaboost_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'adaboost_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("AdaBoost Classifier training completed successfully!")
    print("=" * 70)
