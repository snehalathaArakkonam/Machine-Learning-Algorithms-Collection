"""
Gaussian Mixture Model (GMM) Implementation
==========================================
A Gaussian Mixture Model is a probabilistic clustering algorithm that represents
the data as a mixture of Gaussian distributions. It uses the Expectation-Maximization
(EM) algorithm to find the parameters of the mixture components.

Algorithm (EM):
1. Initialize K Gaussian components with random parameters
2. Expectation Step: Calculate posterior probability of each sample belonging to each component
3. Maximization Step: Update component parameters based on posterior probabilities
4. Repeat until convergence

Key Concepts:
- Soft clustering: samples have probability distribution over clusters
- Mixture coefficients: weight of each Gaussian component
- Component parameters: mean and covariance of each Gaussian
- Log-likelihood: measure of model fit

Advantages:
- Probabilistic framework
- Soft assignment (posterior probabilities)
- Can determine optimal number of components using BIC/AIC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    \"\"\"
    Gaussian Mixture Model using EM algorithm.
    
    Parameters:
    -----------
    n_components : int, default=3
        Number of Gaussian components
    max_iterations : int, default=100
        Maximum iterations for EM algorithm
    tolerance : float, default=1e-3
        Convergence tolerance
    random_state : int, default=None
        Random seed
    \"\"\"
    
    def __init__(self, n_components=3, max_iterations=100, tolerance=1e-3, random_state=None):
        \"\"\"Initialize GMM parameters.\"\"\"
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        self.weights = None
        self.means = None
        self.covariances = None
        self.responsibilities = None
        self.log_likelihood_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X):
        \"\"\"
        Fit GMM using EM algorithm.
        
        Algorithm:
        Step 1: Initialize parameters
        Step 2: E-step: Calculate responsibilities (posterior probabilities)
        Step 3: M-step: Update component parameters
        Step 4: Calculate log-likelihood
        Step 5: Check convergence
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : fitted model
        \"\"\"
        n_samples, n_features = X.shape
        
        # Step 1: Initialize parameters
        # Random initialization of means
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices].copy()
        
        # Initialize covariances as identity matrices scaled by variance
        self.covariances = np.array([np.eye(n_features) * np.var(X) for _ in range(self.n_components)])
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
        print(f\"Fitting GMM with {self.n_components} components using EM algorithm...\")
        
        # EM iterations
        for iteration in range(self.max_iterations):
            # Step 2: E-step - Calculate responsibilities
            self.responsibilities = self._expectation_step(X)
            
            # Step 3: M-step - Update parameters
            self._maximization_step(X)
            
            # Step 4: Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Step 5: Check convergence
            if iteration > 0:
                ll_diff = abs(log_likelihood - self.log_likelihood_history[-2])
                if ll_diff < self.tolerance:
                    print(f\"  Converged at iteration {iteration}\")
                    break
            
            if (iteration + 1) % 20 == 0:
                print(f\"  Iteration {iteration + 1}, Log-likelihood: {log_likelihood:.4f}\")
        
        print(f\"✓ Fitting complete! Final log-likelihood: {log_likelihood:.4f}\")
        
        return self
    
    def _expectation_step(self, X):
        \"\"\"
        E-step: Calculate posterior probabilities (responsibilities).
        
        Responsibility = (weight * likelihood) / total likelihood
        \"\"\"
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Multivariate Gaussian PDF
            pdf = multivariate_normal.pdf(X, self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * pdf
        
        # Normalize
        total_likelihood = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= (total_likelihood + 1e-10)
        
        return responsibilities
    
    def _maximization_step(self, X):
        \"\"\"
        M-step: Update component parameters based on responsibilities.
        \"\"\"
        n_samples = X.shape[0]
        
        # Calculate effective number of samples for each component
        N_k = np.sum(self.responsibilities, axis=0)
        
        # Update weights
        self.weights = N_k / n_samples
        
        # Update means
        for k in range(self.n_components):
            self.means[k] = np.sum(self.responsibilities[:, k:k+1] * X, axis=0) / (N_k[k] + 1e-10)
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_cov = np.dot((self.responsibilities[:, k:k+1] * diff).T, diff) / (N_k[k] + 1e-10)
            
            # Add regularization to ensure positive definiteness
            self.covariances[k] = weighted_cov + 1e-6 * np.eye(X.shape[1])
    
    def _calculate_log_likelihood(self, X):
        \"\"\"Calculate log-likelihood of data under model.\"\"\"
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            likelihood = 0
            for k in range(self.n_components):
                pdf = multivariate_normal.pdf(X[i], self.means[k], self.covariances[k])
                likelihood += self.weights[k] * pdf
            
            log_likelihood += np.log(likelihood + 1e-10)
        
        return log_likelihood
    
    def predict(self, X):
        \"\"\"
        Assign samples to clusters (hard assignment).
        
        Each sample assigned to component with highest responsibility.
        \"\"\"
        responsibilities = self._expectation_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        \"\"\"
        Return posterior probabilities (soft assignment).
        \"\"\"
        return self._expectation_step(X)


# ============================================================================
# SAMPLE USAGE: Gaussian Mixture Model
# ============================================================================
if __name__ == \"__main__\":
    print(\"=\" * 70)
    print(\"GAUSSIAN MIXTURE MODEL (GMM) - SAMPLE USAGE\")
    print(\"=\" * 70)
    
    # Step 1: Create synthetic dataset
    print(\"\\n[Step 1] Creating synthetic clustering dataset...\")
    X, y_true = make_blobs(n_samples=300, n_features=2, centers=3,
                          cluster_std=0.8, random_state=42)
    
    print(f\"Dataset shape: {X.shape}\")
    print(f\"True number of clusters: {len(np.unique(y_true))}\")
    
    # Step 2: Standardize features
    print(\"\\n[Step 2] Standardizing features...\")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 3: Fit GMM
    print(\"\\n[Step 3] Fitting Gaussian Mixture Model...\")
    gmm = GaussianMixtureModel(n_components=3, max_iterations=100, random_state=42)
    gmm.fit(X_scaled)
    
    # Step 4: Get cluster assignments
    print(\"\\n[Step 4] Getting cluster assignments...\")
    labels = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    print(f\"Cluster labels: {np.unique(labels)}\")
    print(f\"Cluster sizes: {np.bincount(labels)}\")
    
    # Step 5: Model parameters
    print(\"\\n[Step 5] Model Parameters:\")
    print(f\"Component weights: {gmm.weights}\")
    print(f\"Component means:\\n{gmm.means}\")
    
    # Step 6: Visualization
    print(\"\\n[Step 6] Generating visualizations...\")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Clustering results
    colors = plt.cm.viridis(np.linspace(0, 1, gmm.n_components))
    
    for k in range(gmm.n_components):
        mask = labels == k
        axes[0, 0].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                          c=[colors[k]], s=100, alpha=0.6,
                          label=f'Component {k}', edgecolors='black', linewidth=1)
    
    # Plot component means
    axes[0, 0].scatter(gmm.means[:, 0], gmm.means[:, 1],
                      c='red', s=300, marker='*', edgecolors='black',
                      linewidth=2, label='Component Means', zorder=5)
    
    axes[0, 0].set_xlabel('Feature 1', fontsize=11)
    axes[0, 0].set_ylabel('Feature 2', fontsize=11)
    axes[0, 0].set_title('GMM Clustering Results', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood evolution
    axes[0, 1].plot(range(len(gmm.log_likelihood_history)), gmm.log_likelihood_history,
                   linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_xlabel('EM Iteration', fontsize=11)
    axes[0, 1].set_ylabel('Log-Likelihood', fontsize=11)
    axes[0, 1].set_title('EM Algorithm Convergence', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Component weights
    axes[1, 0].bar(range(gmm.n_components), gmm.weights,
                  color='steelblue', edgecolor='black', linewidth=2)
    axes[1, 0].set_xlabel('Component ID', fontsize=11)
    axes[1, 0].set_ylabel('Weight', fontsize=11)
    axes[1, 0].set_title('Component Mixture Weights', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Probability heatmap for first 20 samples
    im = axes[1, 1].imshow(probabilities[:20].T, cmap='hot', aspect='auto')
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_ylabel('Component ID', fontsize=11)
    axes[1, 1].set_title('Posterior Probabilities (First 20 Samples)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='Probability')
    
    plt.tight_layout()
    plt.savefig('gmm_visualization.png', dpi=300, bbox_inches='tight')
    print(\"✓ Visualization saved as 'gmm_visualization.png'\")
    plt.show()
    
    print(\"\\n\" + \"=\" * 70)
    print(\"Gaussian Mixture Model training completed successfully!\")
    print(\"=\" * 70)
