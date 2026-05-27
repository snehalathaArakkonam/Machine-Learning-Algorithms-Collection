"""
Gaussian Naive Bayes Classifier Implementation
==============================================
Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem with
the assumption that features follow a Gaussian (normal) distribution.

Mathematical Foundation:
P(y|X) = P(X|y) * P(y) / P(X)

Naive Assumption: Features are conditionally independent given the class label.
This simplifies computation even though the assumption is often violated.

Gaussian Assumption: Each feature distribution per class is Gaussian:
P(x_i|y) = (1/sqrt(2π*σ_y²)) * exp(-(x_i - μ_y)² / (2*σ_y²))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier Implementation.

    Assumes that features follow a Gaussian distribution within each class.

    Attributes:
    -----------
    class_priors : dict
        Prior probability P(y) for each class
    mean : dict
        Mean of each feature for each class
    variance : dict
        Variance of each feature for each class
    """

    def __init__(self):
        """Initialize Gaussian Naive Bayes."""
        self.class_priors = None
        self.mean = None
        self.variance = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes model by calculating mean and variance.

        Step 1: Identify unique classes
        Step 2: For each class:
            - Calculate prior probability: P(y) = count(y) / total_samples
            - For each feature:
                * Calculate mean: μ = sum(x_i) / n
                * Calculate variance: σ² = sum((x_i - μ)²) / n
        Step 3: Store these parameters for prediction

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        """
        n_samples, n_features = X.shape

        # Step 1: Identify unique classes
        self.classes = np.unique(y)

        # Step 2: Calculate parameters for each class
        self.mean = {}
        self.variance = {}
        self.class_priors = {}

        for cls in self.classes:
            # Get all samples of this class
            X_cls = X[y == cls]

            # Step 2a: Prior probability
            self.class_priors[cls] = len(X_cls) / n_samples

            # Step 2b: Mean for each feature
            self.mean[cls] = np.mean(X_cls, axis=0)

            # Step 2c: Variance for each feature
            self.variance[cls] = np.var(X_cls, axis=0)

        print(
            f"Gaussian Naive Bayes fitted on {n_samples} samples with {n_features} features"
        )
        print(f"Classes: {self.classes}")

        return self

    def predict(self, X):
        """
        Predict class labels using Gaussian Naive Bayes.

        Algorithm:
        Step 1: For each sample and each class:
            - Calculate P(y) = prior probability
            - For each feature:
                * Calculate P(x_i|y) using Gaussian distribution
                * Multiply all feature probabilities: P(X|y) = Π P(x_i|y)
            - Calculate P(y|X) = P(X|y) * P(y)
        Step 2: Select class with highest probability

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if self.mean is None:
            raise ValueError("Model must be fit before predictions.")

        predictions = []

        for sample in X:
            # Calculate posterior probability for each class
            posteriors = {}

            for cls in self.classes:
                # Start with prior probability
                prior = np.log(self.class_priors[cls])

                # Calculate likelihood: P(X|y) using Gaussian PDF
                likelihood = 0
                for i, value in enumerate(sample):
                    mean = self.mean[cls][i]
                    variance = self.variance[cls][i]

                    # Gaussian PDF: avoid log(0) by using log-likelihood
                    numerator = (value - mean) ** 2
                    denominator = 2 * variance

                    # log(P(x_i|y))
                    likelihood += (
                        -0.5 * np.log(2 * np.pi * variance) - numerator / denominator
                    )

                # Posterior: log(P(y|X)) = log(P(X|y)) + log(P(y))
                posterior = prior + likelihood
                posteriors[cls] = posterior

            # Select class with maximum posterior
            prediction = max(posteriors, key=posteriors.get)
            predictions.append(prediction)

        return np.array(predictions)

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
        if self.mean is None:
            raise ValueError("Model must be fit before predictions.")

        probabilities = []

        for sample in X:
            posteriors = {}

            for cls in self.classes:
                prior = np.log(self.class_priors[cls])
                likelihood = 0

                for i, value in enumerate(sample):
                    mean = self.mean[cls][i]
                    variance = self.variance[cls][i]
                    numerator = (value - mean) ** 2
                    denominator = 2 * variance
                    likelihood += (
                        -0.5 * np.log(2 * np.pi * variance) - numerator / denominator
                    )

                posteriors[cls] = prior + likelihood

            # Convert log-probabilities to probabilities
            # Subtract max for numerical stability
            max_post = max(posteriors.values())
            scaled_posts = {k: v - max_post for k, v in posteriors.items()}
            exp_posts = {k: np.exp(v) for k, v in scaled_posts.items()}
            sum_exp = sum(exp_posts.values())
            proba = [exp_posts.get(cls, 0) / sum_exp for cls in self.classes]
            probabilities.append(proba)

        return np.array(probabilities)


# ============================================================================
# SAMPLE USAGE: Gaussian Naive Bayes on Iris Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GAUSSIAN NAIVE BAYES CLASSIFIER - SAMPLE USAGE")
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

    # Step 3: Create and train Gaussian Naive Bayes
    print("\n[Step 3] Training Gaussian Naive Bayes Classifier...")
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Step 4: Make predictions
    print("\n[Step 4] Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba = model.predict_proba(X_test[:10])

    # Step 5: Calculate accuracy
    print("\n[Step 5] Calculating accuracy metrics...")
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({int(train_accuracy*100)}%)")
    print(f"Testing Accuracy:  {test_accuracy:.4f} ({int(test_accuracy*100)}%)")

    # Step 6: Per-class accuracy
    print("\n[Step 6] Per-class Accuracy (Test Set):")
    print(f"{'Class':<15} {'Accuracy':<15}")
    print("-" * 30)

    for i in range(len(np.unique(y))):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred_test[mask] == y_test[mask])
            print(f"{iris.target_names[i]:<15} {class_accuracy:<15.4f}")

    # Step 7: Sample predictions with probabilities
    print(
        "\n[Step 7] Sample Predictions with Class Probabilities (First 10 test samples):"
    )
    print(f"{'Actual':<15} {'Predicted':<15} {'Confidence':<12} {'Correct':<8}")
    print("-" * 50)

    for i in range(min(10, len(y_test))):
        max_proba = np.max(y_proba[i])
        actual = iris.target_names[y_test[i]]
        predicted = iris.target_names[y_pred_test[i]]
        is_correct = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(f"{actual:<15} {predicted:<15} {max_proba:<12.2%} {is_correct:<8}")

    # Step 8: Confusion matrix
    print("\n[Step 8] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    # Step 9: Model parameters (mean and variance for each class)
    print("\n[Step 9] Model Parameters (Mean for each class):")
    for cls in model.classes:
        print(f"\nClass {cls} ({iris.target_names[cls]}):")
        print(f"  Prior probability: {model.class_priors[cls]:.4f}")
        for feat_idx, feat_name in enumerate(iris.feature_names):
            mean_val = model.mean[cls][feat_idx]
            var_val = model.variance[cls][feat_idx]
            print(f"  {feat_name}: μ={mean_val:.4f}, σ²={var_val:.4f}")

    # Step 10: Visualization
    print("\n[Step 10] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Confusion matrix
    im = axes[0, 0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0, 0].set_xlabel("Predicted Label", fontsize=11)
    axes[0, 0].set_ylabel("True Label", fontsize=11)
    axes[0, 0].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(range(len(np.unique(y))))
    axes[0, 0].set_yticks(range(len(np.unique(y))))

    for i in range(len(np.unique(y))):
        for j in range(len(np.unique(y))):
            text = axes[0, 0].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    # Plot 2: Per-class accuracy
    class_accuracies = []
    class_names = []
    for i in range(len(np.unique(y))):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.mean(y_pred_test[mask] == y_test[mask])
            class_accuracies.append(acc)
            class_names.append(iris.target_names[i])

    bars = axes[0, 1].barh(
        class_names,
        class_accuracies,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
        edgecolor="black",
        linewidth=2,
    )
    axes[0, 1].set_xlabel("Accuracy", fontsize=11)
    axes[0, 1].set_title("Per-Class Accuracy", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlim([0, 1])

    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 1].text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.2%}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Plot 3: Feature distributions (first feature for each class)
    for cls in model.classes:
        X_cls = X_train[y_train == cls, 0]
        axes[1, 0].hist(
            X_cls, alpha=0.6, label=iris.target_names[cls], bins=15, edgecolor="black"
        )

    axes[1, 0].set_xlabel(iris.feature_names[0], fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title(
        "Feature Distribution by Class", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Training vs Testing accuracy
    accuracies_compare = [train_accuracy, test_accuracy]
    datasets = ["Training", "Testing"]
    colors = ["skyblue", "orange"]
    bars = axes[1, 1].bar(
        datasets, accuracies_compare, color=colors, edgecolor="black", linewidth=2
    )
    axes[1, 1].set_ylabel("Accuracy", fontsize=11)
    axes[1, 1].set_title(
        "Gaussian Naive Bayes: Train vs Test", fontsize=12, fontweight="bold"
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
    plt.savefig("gaussian_naive_bayes_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'gaussian_naive_bayes_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Gaussian Naive Bayes training completed successfully!")
    print("=" * 70)
