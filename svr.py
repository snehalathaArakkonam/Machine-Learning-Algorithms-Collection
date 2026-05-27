"""
Support Vector Regression (SVR) Implementation
==============================================
SVR is a regression version of SVM that finds the optimal hyperplane to fit data
with maximum margin. Unlike SVM for classification, SVR minimizes the prediction
error while maintaining a margin of tolerance around predictions.

Key Concepts:
1. Epsilon-insensitive loss: Errors within epsilon margin are ignored
2. Support vectors: Points outside the epsilon margin or on the boundary
3. Regularization parameter C: Controls trade-off between margin and error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SupportVectorRegression:
    """
    Support Vector Regression (SVR) Implementation.

    Parameters:
    -----------
    C : float, default=1.0
        Regularization strength. Higher C = stricter training
    epsilon : float, default=0.1
        Epsilon tube: errors within this margin are ignored
    kernel : str, default='linear'
        Kernel type ('linear', 'rbf', 'poly')
    gamma : float, default=0.1
        Kernel coefficient for 'rbf' and 'poly'
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
        epsilon=0.1,
        kernel="linear",
        gamma=0.1,
        degree=3,
        learning_rate=0.001,
        max_iterations=1000,
    ):
        """Initialize SVR parameters."""
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train SVR using gradient descent with epsilon-insensitive loss.

        Step 1: Initialize weights and bias
        Step 2: For each iteration:
            - Compute predictions
            - Calculate epsilon-insensitive loss
            - Update weights using gradient descent
        Step 3: Identify support vectors

        Epsilon-Insensitive Loss:
        L_ε(y, y_pred) = max(0, |y - y_pred| - ε)

        Total Loss = MSE_epsilon + C * L2_regularization

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training target values
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.X_train = X
        self.y_train = y

        print(f"Training SVR with {self.kernel} kernel...")
        print(f"Parameters: C={self.C}, epsilon={self.epsilon}")

        # Step 2: Optimize using gradient descent
        for iteration in range(self.max_iterations):
            # Make predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute residuals
            residuals = y - y_pred

            # Epsilon-insensitive loss: max(0, |residual| - epsilon)
            losses = np.maximum(0, np.abs(residuals) - self.epsilon)

            # Mean epsilon-insensitive loss
            mean_loss = np.mean(losses)

            # L2 regularization term
            l2_penalty = (self.C / 2) * np.sum(self.weights**2)

            # Total loss
            total_loss = mean_loss + l2_penalty

            # Compute gradients
            # Gradient of epsilon-insensitive loss
            dw = np.zeros_like(self.weights)
            db = 0

            for i in range(n_samples):
                if np.abs(residuals[i]) > self.epsilon:
                    # Point is outside epsilon tube
                    sign = np.sign(residuals[i])
                    dw += -sign * X[i]
                    db += -sign

            # Normalize gradients and add L2 penalty
            dw = dw / n_samples + (self.C / n_samples) * self.weights
            db = db / n_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (iteration + 1) % 200 == 0:
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}, Loss: {total_loss:.4f}"
                )

        print("✓ Training completed!")

        return self

    def predict(self, X):
        """
        Predict continuous values for samples.

        Mathematical Formula:
        y_pred = X * weights + bias

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fit before predictions.")

        return np.dot(X, self.weights) + self.bias


# ============================================================================
# SAMPLE USAGE: SVR on Diabetes Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SUPPORT VECTOR REGRESSION (SVR) - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load diabetes dataset
    print("\n[Step 1] Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print(f"Dataset shape: {X.shape}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

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

    # Step 4: Train SVR
    print("\n[Step 4] Training Support Vector Regression (Linear kernel)...")
    model = SupportVectorRegression(
        C=100.0, epsilon=5.0, kernel="linear", learning_rate=0.001, max_iterations=1000
    )
    model.fit(X_train_scaled, y_train)

    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Step 6: Calculate performance metrics
    print("\n[Step 6] Calculating performance metrics...")

    # Mean Absolute Error (MAE)
    mae_train = np.mean(np.abs(y_pred_train - y_train))
    mae_test = np.mean(np.abs(y_pred_test - y_test))

    # Mean Squared Error (MSE)
    mse_train = np.mean((y_pred_train - y_train) ** 2)
    mse_test = np.mean((y_pred_test - y_test) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    # R-squared score
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print("-" * 50)
    print(f"{'MAE':<20} {mae_train:<15.4f} {mae_test:<15.4f}")
    print(f"{'MSE':<20} {mse_train:<15.4f} {mse_test:<15.4f}")
    print(f"{'RMSE':<20} {rmse_train:<15.4f} {rmse_test:<15.4f}")
    print(f"{'R² Score':<20} {' ':<15} {r2_score:<15.4f}")

    # Step 7: Model coefficients
    print("\n[Step 7] Model Coefficients:")
    for i, coef in enumerate(model.weights):
        print(f"  Feature {i}: {coef:.6f}")
    print(f"  Bias: {model.bias:.6f}")

    # Step 8: Sample predictions
    print("\n[Step 8] Sample Predictions (First 10 test samples):")
    print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12} {'% Error':<12}")
    print("-" * 48)

    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        pct_error = (error / y_test[i]) * 100 if y_test[i] != 0 else 0
        print(
            f"{y_test[i]:<12.2f} {y_pred_test[i]:<12.2f} {error:<12.2f} {pct_error:<12.1f}%"
        )

    # Step 9: Visualization
    print("\n[Step 9] Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Actual vs Predicted (Test Set)
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="k", s=50)
    axes[0, 0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0, 0].set_xlabel("Actual Values", fontsize=11)
    axes[0, 0].set_ylabel("Predicted Values", fontsize=11)
    axes[0, 0].set_title(
        "SVR: Actual vs Predicted (Test Set)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6, edgecolors="k", s=50)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].axhline(
        y=model.epsilon, color="g", linestyle=":", lw=2, label="Epsilon margin"
    )
    axes[0, 1].axhline(y=-model.epsilon, color="g", linestyle=":", lw=2)
    axes[0, 1].set_xlabel("Predicted Values", fontsize=11)
    axes[0, 1].set_ylabel("Residuals", fontsize=11)
    axes[0, 1].set_title("Residuals vs Predictions", fontsize=12, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Error distribution
    axes[1, 0].hist(residuals, bins=20, edgecolor="black", color="skyblue", alpha=0.7)
    axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Residual Value", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Performance metrics comparison
    metrics_names = ["MAE", "RMSE"]
    train_metrics = [mae_train, rmse_train]
    test_metrics = [mae_test, rmse_test]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[1, 1].bar(
        x - width / 2,
        train_metrics,
        width,
        label="Training",
        color="skyblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = axes[1, 1].bar(
        x + width / 2,
        test_metrics,
        width,
        label="Testing",
        color="orange",
        edgecolor="black",
        linewidth=1.5,
    )

    axes[1, 1].set_ylabel("Error Value", fontsize=11)
    axes[1, 1].set_title("Performance Metrics", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("svr_visualization.png", dpi=300, bbox_inches="tight")
    print("✓ Visualization saved as 'svr_visualization.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("Support Vector Regression training completed successfully!")
    print("=" * 70)
