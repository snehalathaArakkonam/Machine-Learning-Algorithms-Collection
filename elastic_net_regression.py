"""
Elastic Net Regression Implementation
======================================
Elastic Net is a linear regression technique that combines L1 (Lasso) and L2 (Ridge) regularization.
It helps prevent overfitting by penalizing both large coefficients and feature selection.

Mathematical Formula:
Cost = MSE + alpha * (l1_ratio * |w| + (1 - l1_ratio) * w²)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ElasticNetRegression:
    """
    Elastic Net Regression: Combines L1 (Lasso) and L2 (Ridge) regularization.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength. Higher values = more regularization.
    l1_ratio : float, default=0.5
        Proportion of L1 penalty (0 = Ridge only, 1 = Lasso only).
    learning_rate : float, default=0.01
        Step size for gradient descent optimization.
    max_iterations : int, default=1000
        Maximum number of iterations for convergence.
    tolerance : float, default=1e-4
        Convergence threshold for early stopping.
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-4,
    ):
        """Initialize Elastic Net Regression parameters."""
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the Elastic Net model using gradient descent.

        Step 1: Initialize weights and bias to zeros
        Step 2: For each iteration, compute predictions
        Step 3: Calculate the combined L1+L2 penalty loss
        Step 4: Update weights using gradient descent with regularization
        Step 5: Stop when convergence is achieved or max iterations reached

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training target values
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        # Step 2-5: Gradient descent optimization
        for iteration in range(self.max_iterations):
            # Make predictions using current weights
            y_pred = X.dot(self.weights) + self.bias

            # Calculate prediction error
            error = y_pred - y

            # Calculate MSE loss
            mse_loss = np.mean(error**2)

            # L1 penalty (Lasso): sum of absolute values of weights
            l1_penalty = np.sum(np.abs(self.weights))

            # L2 penalty (Ridge): sum of squared weights
            l2_penalty = np.sum(self.weights**2)

            # Combined Elastic Net loss
            total_loss = mse_loss + self.alpha * (
                self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty
            )

            self.loss_history.append(total_loss)

            # Check for convergence
            if (
                iteration > 0
                and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance
            ):
                print(f"Converged at iteration {iteration}")
                break

            # Gradient of MSE
            dw_mse = (2 / n_samples) * X.T.dot(error)
            db_mse = (2 / n_samples) * np.sum(error)

            # Gradient of L1 penalty (subgradient)
            dw_l1 = np.sign(self.weights)

            # Gradient of L2 penalty
            dw_l2 = 2 * self.weights

            # Combined gradient
            dw = dw_mse + self.alpha * (
                self.l1_ratio * dw_l1 + (1 - self.l1_ratio) * dw_l2
            )
            db = db_mse

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Mathematical Formula:
        y_pred = X * weights + bias

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features for prediction

        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fit before predictions can be made.")

        return X.dot(self.weights) + self.bias

    def get_coefficients(self):
        """Return the learned coefficients."""
        return self.weights if self.weights is not None else None


# ============================================================================
# SAMPLE USAGE: Elastic Net Regression on Diabetes Dataset
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ELASTIC NET REGRESSION - SAMPLE USAGE")
    print("=" * 70)

    # Step 1: Load the diabetes dataset
    print("\n[Step 1] Loading diabetes dataset...")
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")

    # Step 2: Split data into training and testing sets
    print("\n[Step 2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Standardize features (important for regularization)
    print("\n[Step 3] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Create and train Elastic Net model
    print("\n[Step 4] Training Elastic Net Regression model...")
    model = ElasticNetRegression(
        alpha=0.1, l1_ratio=0.5, learning_rate=0.01, max_iterations=1000
    )
    model.fit(X_train_scaled, y_train)

    # Step 5: Make predictions
    print("\n[Step 5] Making predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Step 6: Calculate performance metrics
    print("\n[Step 6] Calculating performance metrics...")
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # R-squared score
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print("-" * 50)
    print(f"{'MSE':<20} {train_mse:<15.4f} {test_mse:<15.4f}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'R² Score (Test)':<20} {r2_score:<15.4f}")

    # Step 7: Show model coefficients
    print("\n[Step 7] Model Coefficients:")
    coefficients = model.get_coefficients()
    for i, coef in enumerate(coefficients):
        print(f"  Feature {i}: {coef:.6f}")

    # Step 8: Visualization
    print("\n[Step 8] Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Predicted (Test Set)
    axes[0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors="k")
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0].set_xlabel("Actual Values", fontsize=12)
    axes[0].set_ylabel("Predicted Values", fontsize=12)
    axes[0].set_title(
        "Elastic Net: Actual vs Predicted (Test Set)", fontsize=13, fontweight="bold"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training Loss over iterations
    axes[1].plot(model.loss_history, linewidth=2, color="steelblue")
    axes[1].set_xlabel("Iteration", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Training Loss over Iterations", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "elastic_net_regression_visualization.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Visualization saved as 'elastic_net_regression_visualization.png'")
    plt.show()

    # Step 9: Show sample predictions
    print("\n[Step 9] Sample Predictions (First 10 test samples):")
    print(f"\n{'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 36)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        print(f"{y_test[i]:<12.2f} {y_pred_test[i]:<12.2f} {error:<12.2f}")

    print("\n" + "=" * 70)
    print("Elastic Net Regression training completed successfully!")
    print("=" * 70)
