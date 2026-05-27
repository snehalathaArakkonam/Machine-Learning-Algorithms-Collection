import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # Only for feature transformation


class PolynomialRegression:
    """
    Polynomial Regression using Linear Regression internally
    """

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        self.degree = degree
        self.model = LinearRegression(
            learning_rate, n_iterations
        )  # Reuse Linear Regression

    def fit(self, X, y):
        # Convert to polynomial features
        poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))
        self.model.fit(X_poly, y)

    def predict(self, X):
        poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))
        return self.model.predict(X_poly)


# ===================== SAMPLE USAGE =====================
if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]) + np.random.normal(0, 5, 10)

    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Polynomial Regression Predictions:", np.round(predictions, 2))
