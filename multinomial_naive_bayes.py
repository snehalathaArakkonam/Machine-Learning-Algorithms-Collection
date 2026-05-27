"""
Multinomial Naive Bayes Classifier Implementation
================================================
Multinomial Naive Bayes is designed for features that represent counts or frequencies.
Commonly used for text classification where features are word counts.

Mathematical Model:
P(y|X) = P(y) * Π P(x_i|y)

For Multinomial: P(x_i|y) = (count_of_feature_i_in_class_y + alpha) / (total_count_in_class_y + alpha*n_features)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes Classifier for count/frequency features.

    Parameters:
    -----------
    alpha : float, default=1.0
        Additive smoothing (Laplace smoothing)
        Prevents zero probabilities for unseen features
    """

    def __init__(self, alpha=1.0):
        """Initialize Multinomial Naive Bayes."""
        self.alpha = alpha
        self.class_priors = None
        self.feature_log_probs = None
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        """
        Fit Multinomial Naive Bayes model.

        Step 1: Identify unique classes
        Step 2: For each class:
            - Calculate prior probability: P(y)
            - For each feature:
                * Calculate log-probability with smoothing
                * Log-probability prevents numerical underflow

        Mathematical Formula:
        P(x_i|y) = (count_i_in_y + alpha) / (total_count_in_y + alpha * n_features)
        log(P(x_i|y)) = log(count_i_in_y + alpha) - log(total_count_in_y + alpha * n_features)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature counts
        y : array-like, shape (n_samples,)
            Training labels
        """
        n_samples, n_features = X.shape
        self.n_features = n_features

        # Step 1: Identify unique classes
        self.classes = np.unique(y)

        # Step 2: Calculate parameters
        self.class_priors = {}
        self.feature_log_probs = {}

        for cls in self.classes:
            # Get all samples of this class
            X_cls = X[y == cls]

            # Prior probability
            self.class_priors[cls] = len(X_cls) / n_samples

            # Feature counts with smoothing
            feature_counts = np.sum(X_cls, axis=0) + self.alpha
            total_count = np.sum(feature_counts)

            # Log-probabilities (prevent numerical underflow)
            self.feature_log_probs[cls] = np.log(feature_counts) - np.log(total_count)

        print(
            f"Multinomial Naive Bayes fitted on {n_samples} samples with {n_features} features"
        )
        print(f"Classes: {self.classes}")

        return self

    def predict(self, X):
        """
        Predict class labels for count features.

        Algorithm:
        Step 1: For each sample and each class:
            - Start with log(P(y))
            - For each feature:
                * Add feature_value * log(P(feature|class))
            - Log-posterior = log(P(y)) + sum(x_i * log(P(x_i|y)))
        Step 2: Select class with highest log-posterior

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Count features

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if self.class_priors is None:
            raise ValueError("Model must be fit before predictions.")

        predictions = []

        for sample in X:
            posteriors = {}

            for cls in self.classes:
                # Log of prior probability
                log_prior = np.log(self.class_priors[cls])

                # Log of likelihood (weighted by feature counts)
                log_likelihood = np.sum(sample * self.feature_log_probs[cls])

                # Log-posterior
                log_posterior = log_prior + log_likelihood
                posteriors[cls] = log_posterior

            # Select class with maximum posterior
            prediction = max(posteriors, key=posteriors.get)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        """Return probability estimates for each class."""
        if self.class_priors is None:
            raise ValueError("Model must be fit before predictions.")

        probabilities = []

        for sample in X:
            posteriors = {}

            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = np.sum(sample * self.feature_log_probs[cls])
                posteriors[cls] = log_prior + log_likelihood

            # Convert to probabilities
            max_post = max(posteriors.values())
            scaled_posts = {k: v - max_post for k, v in posteriors.items()}
            exp_posts = {k: np.exp(v) for k, v in scaled_posts.items()}
            sum_exp = sum(exp_posts.values())
            proba = [exp_posts.get(cls, 0) / sum_exp for cls in self.classes]
            probabilities.append(proba)

        return np.array(probabilities)


# ============================================================================
# SAMPLE USAGE: Multinomial Naive Bayes with Count Features
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MULTINOMIAL NAIVE BAYES CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create synthetic count dataset
    print("\n[Step 1] Creating synthetic count feature dataset...")
    np.random.seed(42)

    n_samples = 200
    n_features = 15

    # Generate count features (word frequencies)
    X = np.random.poisson(lam=2.0, size=(n_samples, n_features))

    # Generate labels based on feature patterns
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > 2) & (X[:, 1] > 1)] = 1
    y[(X[:, 2] > 3) & (X[:, 3] > 2)] = 1

    print(f"Dataset shape: {X.shape}")
    print(f"Feature value range: {X.min()} to {X.max()}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Feature statistics - Mean: {X.mean():.2f}, Std: {X.std():.2f}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train Multinomial Naive Bayes
    print("\n[Step 3] Training Multinomial Naive Bayes...")
    model = MultinomialNaiveBayes(alpha=1.0)
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

    # Step 6: Confusion matrix
    print("\n[Step 6] Confusion Matrix (Test Set):")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred_test)
    print(f"               Predicted 0  Predicted 1")
    print(f"Actual 0:      {cm[0, 0]:>11}  {cm[0, 1]:>11}")
    print(f"Actual 1:      {cm[1, 0]:>11}  {cm[1, 1]:>11}")

    # Step 7: Model parameters
    print("\n[Step 7] Model Parameters (Log-Probabilities):")
    for cls in model.classes:
        print(f"\nClass {cls}:")
        print(f"  Prior probability: {model.class_priors[cls]:.4f}")
        print(f"  Top 5 most probable features:")
        top_indices = np.argsort(model.feature_log_probs[cls])[-5:][::-1]
        for idx in top_indices:
            prob = np.exp(model.feature_log_probs[cls][idx])
            print(
                f"    Feature {idx}: log_prob={model.feature_log_probs[cls][idx]:.4f}, prob={prob:.4f}"
            )

    # Step 8: Sample predictions
    print("\n[Step 8] Sample Predictions (First 10 test samples):")
    print(f"{'Actual':<10} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 34)

    for i in range(min(10, len(y_test))):
        max_proba = np.max(y_proba[i])
        is_correct = "✓" if y_test[i] == y_pred_test[i] else "✗"
        print(f"{y_test[i]:<10} {y_pred_test[i]:<12} {max_proba:<12.2%} {is_correct}")

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Confusion matrix
    im = axes[0, 0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0, 0].set_xlabel("Predicted Label", fontsize=11)
    axes[0, 0].set_ylabel("True Label", fontsize=11)
    axes[0, 0].set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
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

    # Plot 2: Feature log-probabilities
    feature_indices = np.arange(min(n_features, 10))

    class_0_logprobs = model.feature_log_probs[0][feature_indices]
    class_1_logprobs = model.feature_log_probs[1][feature_indices]

    x = np.arange(len(feature_indices))
    width = 0.35

    bars1 = axes[0, 1].bar(
        x - width / 2,
        class_0_logprobs,
        width,
        label="Class 0",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[0, 1].bar(
        x + width / 2,
        class_1_logprobs,
        width,
        label="Class 1",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[0, 1].set_xlabel("Feature Index", fontsize=11)
    axes[0, 1].set_ylabel("Log-Probability", fontsize=11)
    axes[0, 1].set_title(
        "Feature Log-Probabilities by Class", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"F{i}" for i in feature_indices])
    axes[0, 1].legend()

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
    axes[1, 0].set_title("Train vs Test Accuracy", fontsize=12, fontweight="bold")
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

    # Plot 4: Feature count distribution by class
    class_0_counts = X[y == 0].mean(axis=0)[:10]
    class_1_counts = X[y == 1].mean(axis=0)[:10]

    x = np.arange(len(class_0_counts))
    width = 0.35

    bars1 = axes[1, 1].bar(
        x - width / 2,
        class_0_counts,
        width,
        label="Class 0",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[1, 1].bar(
        x + width / 2,
        class_1_counts,
        width,
        label="Class 1",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[1, 1].set_xlabel("Feature Index", fontsize=11)
    axes[1, 1].set_ylabel("Average Count", fontsize=11)
    axes[1, 1].set_title(
        "Average Feature Counts by Class", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"F{i}" for i in range(len(class_0_counts))])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        "multinomial_naive_bayes_visualization.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Visualization saved as 'multinomial_naive_bayes_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Multinomial Naive Bayes training completed successfully!")
    print("=" * 70)
