"""
Bernoulli Naive Bayes Classifier Implementation
===============================================
Bernoulli Naive Bayes is designed for binary features (0/1 or boolean values).
It's commonly used for text classification with presence/absence of features.

Mathematical Model:
P(y|X) = P(y) * Π P(x_i|y)

For Bernoulli: P(x_i|y) = p * x_i + (1-p) * (1-x_i)
where p is the probability of feature i being 1 in class y.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes Classifier for binary features.

    Parameters:
    -----------
    alpha : float, default=1.0
        Additive smoothing parameter (Laplace smoothing)
        Prevents zero probabilities
    """

    def __init__(self, alpha=1.0):
        """Initialize Bernoulli Naive Bayes."""
        self.alpha = alpha
        self.class_priors = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes model.

        Step 1: Identify unique classes
        Step 2: For each class:
            - Calculate prior probability: P(y) = count(y) / total_samples
            - For each feature:
                * Calculate probability of feature being 1: P(x_i=1|y)
                * Add Laplace smoothing to avoid zero probabilities

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Binary features (0 or 1)
        y : array-like, shape (n_samples,)
            Training labels
        """
        n_samples, n_features = X.shape

        # Step 1: Identify unique classes
        self.classes = np.unique(y)

        # Step 2: Calculate parameters
        self.class_priors = {}
        self.feature_probs = {}

        for cls in self.classes:
            # Get all samples of this class
            X_cls = X[y == cls]

            # Prior probability
            self.class_priors[cls] = len(X_cls) / n_samples

            # Feature probabilities (probability of being 1)
            # With Laplace smoothing: (count + alpha) / (n_samples + 2*alpha)
            feature_counts = np.sum(X_cls, axis=0)
            self.feature_probs[cls] = (feature_counts + self.alpha) / (
                len(X_cls) + 2 * self.alpha
            )

        print(
            f"Bernoulli Naive Bayes fitted on {n_samples} samples with {n_features} features"
        )
        print(f"Classes: {self.classes}")

        return self

    def predict(self, X):
        """
        Predict class labels for binary features.

        Algorithm:
        Step 1: For each sample and each class:
            - Start with log(P(y))
            - For each feature:
                * If feature=1: add log(P(x_i=1|y))
                * If feature=0: add log(P(x_i=0|y)) = log(1 - P(x_i=1|y))
            - Log-posterior = log(P(y)) + sum of log-likelihoods
        Step 2: Select class with highest log-posterior

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Binary samples

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

                # Log of likelihood
                log_likelihood = 0

                for i, value in enumerate(sample):
                    prob_1 = self.feature_probs[cls][i]
                    prob_0 = 1 - prob_1

                    if value == 1:
                        log_likelihood += np.log(prob_1 + 1e-10)
                    else:
                        log_likelihood += np.log(prob_0 + 1e-10)

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
                log_likelihood = 0

                for i, value in enumerate(sample):
                    prob_1 = self.feature_probs[cls][i]
                    prob_0 = 1 - prob_1

                    if value == 1:
                        log_likelihood += np.log(prob_1 + 1e-10)
                    else:
                        log_likelihood += np.log(prob_0 + 1e-10)

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
# SAMPLE USAGE: Bernoulli Naive Bayes
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("BERNOULLI NAIVE BAYES CLASSIFIER - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Create synthetic binary dataset
    print("\n[Step 1] Creating synthetic binary dataset...")
    np.random.seed(42)

    n_samples = 300
    n_features = 10

    # Generate binary features
    X = np.random.binomial(1, 0.5, (n_samples, n_features))
    # Generate labels based on feature patterns
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] == 1) & (X[:, 1] == 1)] = 1
    y[(X[:, 2] == 1) & (X[:, 3] == 0)] = 1

    print(f"Dataset shape: {X.shape}")
    print(f"Binary features: {np.unique(X)}")
    print(f"Class distribution: {np.bincount(y)}")

    # Step 2: Split data
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train Bernoulli Naive Bayes
    print("\n[Step 3] Training Bernoulli Naive Bayes...")
    model = BernoulliNaiveBayes(alpha=1.0)
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
    print("\n[Step 7] Model Parameters:")
    for cls in model.classes:
        print(f"\nClass {cls}:")
        print(f"  Prior probability: {model.class_priors[cls]:.4f}")
        print(f"  Feature probabilities (P(feature=1|class)):")
        for feat_idx in range(min(5, len(model.feature_probs[cls]))):
            prob = model.feature_probs[cls][feat_idx]
            print(f"    Feature {feat_idx}: {prob:.4f}")

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

    # Plot 2: Feature probabilities for each class
    feature_indices = np.arange(min(n_features, 8))
    width = 0.35

    class_0_probs = model.feature_probs[0][feature_indices]
    class_1_probs = model.feature_probs[1][feature_indices]

    x = np.arange(len(feature_indices))
    bars1 = axes[0, 1].bar(
        x - width / 2,
        class_0_probs,
        width,
        label="Class 0",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[0, 1].bar(
        x + width / 2,
        class_1_probs,
        width,
        label="Class 1",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[0, 1].set_xlabel("Feature Index", fontsize=11)
    axes[0, 1].set_ylabel("P(feature=1|class)", fontsize=11)
    axes[0, 1].set_title(
        "Feature Probabilities by Class", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"F{i}" for i in feature_indices])
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])

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

    # Plot 4: Class distribution
    train_dist = np.bincount(y_train)
    test_dist = np.bincount(y_test)
    classes = ["Class 0", "Class 1"]

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
    plt.savefig("bernoulli_naive_bayes_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'bernoulli_naive_bayes_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Bernoulli Naive Bayes training completed successfully!")
    print("=" * 70)
