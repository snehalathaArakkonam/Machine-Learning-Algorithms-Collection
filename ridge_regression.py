import numpy as np


class RidgeRegression:
    """
    Ridge Regression (L2 Regularization)
    Prevents overfitting by adding penalty on weights
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        self.alpha = alpha  # Regularization strength
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        n_samples = X.shape[0]

        self.weights = np.zeros(1)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (
                2 * self.alpha * self.weights
            )
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias
