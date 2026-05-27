"""
Mean Shift Clustering Implementation
===================================
Mean Shift is a non-parametric, density-based clustering algorithm that finds
the modes of the probability density function and groups data points around them.

Algorithm:
1. Initialize window (bandwidth) for kernel density estimation
2. For each data point:
    a. Calculate kernel density estimate at current position
    b. Shift to the mean of points within kernel window
    c. Repeat until convergence
3. Group points that converge to same mode

Key Concepts:
- Kernel: weighting function (typically Gaussian or flat kernel)
- Bandwidth: radius of the kernel window
- Density peaks: modes where points converge
- No need to specify number of clusters beforehand

Advantages:
- Automatically determines number of clusters
- Handles arbitrary-shaped clusters
- Robust to outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class MeanShift:
    \"\"\"
    Mean Shift Clustering Algorithm.
    
    Parameters:
    -----------
    bandwidth : float, default=1.0
        Bandwidth of the kernel
    kernel : str, default='gaussian'
        Kernel type: 'gaussian' or 'flat'
    max_iterations : int, default=100
        Maximum iterations for convergence
    tolerance : float, default=1e-3
        Convergence tolerance
    \"\"\"
    
    def __init__(self, bandwidth=1.0, kernel='gaussian', max_iterations=100, tolerance=1e-3):
        \"\"\"Initialize Mean Shift parameters.\"\"\"
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cluster_centers = None
        self.labels = None
    
    def fit(self, X):
        \"\"\"
        Perform Mean Shift clustering.
        
        Algorithm:
        Step 1: Initialize window positions (copy of input data)
        Step 2: For each point, shift until convergence:
            a. Calculate kernel weights
            b. Compute weighted mean
            c. Move to weighted mean
            d. Check for convergence
        Step 3: Group converged points to same mode
        Step 4: Return cluster labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        self : fitted model
        \"\"\"
        n_samples = X.shape[0]
        
        # Step 1: Initialize with copy of data points
        current_positions = X.copy().astype(float)
        initial_positions = X.copy()
        
        print(f\"Performing Mean Shift clustering with bandwidth={self.bandwidth}...\")
        
        # Step 2: Shift each point until convergence
        for iteration in range(self.max_iterations):
            shifts = np.zeros_like(current_positions)
            
            for i in range(n_samples):
                # Calculate distances to all points
                distances = np.linalg.norm(X - current_positions[i], axis=1)
                
                # Kernel weighting
                if self.kernel == 'gaussian':
                    weights = np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2))
                else:  # flat kernel
                    weights = (distances <= self.bandwidth).astype(float)
                
                # Weighted mean (shift vector)
                weighted_sum = np.sum(weights[:, np.newaxis] * X, axis=0)
                weight_sum = np.sum(weights)
                
                if weight_sum > 0:
                    shifts[i] = weighted_sum / weight_sum - current_positions[i]
            
            # Update positions
            current_positions += shifts
            
            # Check convergence
            max_shift = np.max(np.abs(shifts))
            if max_shift < self.tolerance:
                print(f\"  Converged at iteration {iteration}\")
                break
            
            if (iteration + 1) % 20 == 0:
                print(f\"  Iteration {iteration + 1}, max_shift={max_shift:.6f}\")
        
        # Step 3: Identify cluster centers by grouping similar positions
        self.cluster_centers = []
        self.labels = np.zeros(n_samples, dtype=int)
        
        cluster_id = 0
        used = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if used[i]:
                continue
            
            # Find all points close to this converged position
            distances = np.linalg.norm(current_positions - current_positions[i], axis=1)
            cluster_mask = distances < 2 * self.bandwidth
            
            self.labels[cluster_mask] = cluster_id
            used[cluster_mask] = True
            
            # Store cluster center
            self.cluster_centers.append(current_positions[i])
            cluster_id += 1
        
        self.cluster_centers = np.array(self.cluster_centers)
        
        print(f\"✓ Clustering complete! Found {len(self.cluster_centers)} clusters\")
        
        return self
    
    def fit_predict(self, X):
        \"\"\"Fit and return cluster labels.\"\"\"
        self.fit(X)
        return self.labels
    
    def predict(self, X_new):
        \"\"\"
        Assign new samples to nearest cluster.
        
        Parameters:
        -----------
        X_new : array-like, shape (n_samples, n_features)
            New data points
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        \"\"\"
        if self.cluster_centers is None:
            raise ValueError(\"Model must be fit before predictions.\")
        
        labels = np.zeros(X_new.shape[0], dtype=int)
        
        for i in range(X_new.shape[0]):
            # Find nearest cluster center
            distances = np.linalg.norm(self.cluster_centers - X_new[i], axis=1)
            labels[i] = np.argmin(distances)
        
        return labels


# ============================================================================
# SAMPLE USAGE: Mean Shift Clustering
# ============================================================================
if __name__ == \"__main__\":
    print(\"=\" * 70)
    print(\"MEAN SHIFT CLUSTERING - SAMPLE USAGE\")
    print(\"=\" * 70)
    
    # Step 1: Create synthetic dataset
    print(\"\\n[Step 1] Creating synthetic clustering dataset...\")
    X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)
    
    print(f\"Dataset shape: {X.shape}\")
    print(f\"True number of clusters: {len(np.unique(y_true))}\")
    
    # Step 2: Standardize features
    print(\"\\n[Step 2] Standardizing features...\")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 3: Fit Mean Shift
    print(\"\\n[Step 3] Fitting Mean Shift...\")
    mean_shift = MeanShift(bandwidth=0.5, kernel='gaussian', max_iterations=100)
    labels = mean_shift.fit_predict(X_scaled)
    
    # Step 4: Analyze results
    print(\"\\n[Step 4] Clustering Results:\")
    print(f\"Number of clusters found: {len(mean_shift.cluster_centers)}\")
    print(f\"Cluster sizes: {np.bincount(labels)}\")
    
    # Step 5: Visualization
    print(\"\\n[Step 5] Generating visualizations...\")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Clustering results
    colors = plt.cm.viridis(np.linspace(0, 1, len(mean_shift.cluster_centers)))
    
    for cluster_id in range(len(mean_shift.cluster_centers)):
        mask = labels == cluster_id
        axes[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                       c=[colors[cluster_id]], s=100, alpha=0.6,
                       label=f'Cluster {cluster_id}', edgecolors='black', linewidth=1)
    
    # Plot cluster centers
    axes[0].scatter(mean_shift.cluster_centers[:, 0],
                   mean_shift.cluster_centers[:, 1],
                   c='red', s=300, marker='*', edgecolors='black',
                   linewidth=2, label='Cluster Centers', zorder=5)
    
    axes[0].set_xlabel('Feature 1', fontsize=11)
    axes[0].set_ylabel('Feature 2', fontsize=11)
    axes[0].set_title('Mean Shift Clustering Results', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cluster size distribution
    cluster_sizes = np.bincount(labels)
    axes[1].bar(range(len(cluster_sizes)), cluster_sizes,
               color='steelblue', edgecolor='black', linewidth=2)
    axes[1].set_xlabel('Cluster ID', fontsize=11)
    axes[1].set_ylabel('Number of Points', fontsize=11)
    axes[1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mean_shift_visualization.png', dpi=300, bbox_inches='tight')
    print(\"✓ Visualization saved as 'mean_shift_visualization.png'\")
    plt.show()
    
    # Step 6: Predict on new data
    print(\"\\n[Step 6] Predicting cluster labels for new data...\")
    X_new = np.array([[0.5, 0.5], [-1.0, -1.0]])
    new_labels = mean_shift.predict(X_new)
    print(f\"New samples assigned to clusters: {new_labels}\")
    
    print(\"\\n\" + \"=\" * 70)
    print(\"Mean Shift Clustering completed successfully!\")
    print(\"=\" * 70)
