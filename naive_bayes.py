"""
Naive Bayes Classifier (Base Implementation)
============================================
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the
assumption that features are conditionally independent given the class label.

Bayes' Theorem:
P(y|X) = P(X|y) * P(y) / P(X)

Naive Independence Assumption:
P(X|y) = Π P(x_i|y)  for all features x_i

This base implementation provides the generalized framework that can handle
different feature distributions (Gaussian, Multinomial, Bernoulli, etc.).

Algorithm:
1. Calculate prior probabilities P(y) from training data
2. Calculate feature likelihoods P(x_i|y) for each feature and class
3. For predictions: use Bayes rule to compute posterior probabilities
4. Assign sample to class with highest posterior probability

Key Advantages:
- Fast and efficient
- Works well with high-dimensional data
- Requires relatively small training sets
- Good baseline classifier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter


class NaiveBayes:
    \"\"\"
    Naive Bayes Classifier (Gaussian variant).
    
    Assumes features follow Gaussian (normal) distribution within each class.
    
    For each class and feature, stores:
    - Mean (μ): average value
    - Variance (σ²): spread of values
    
    Then uses Gaussian PDF to calculate likelihoods.
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize Naive Bayes classifier.\"\"\"
        self.classes = None
        self.priors = None           # Prior probabilities P(y)
        self.means = None            # Feature means per class
        self.variances = None        # Feature variances per class
        self.eps = 1e-9              # Small value to avoid division by zero
    
    def fit(self, X, y):
        \"\"\"
        Train Naive Bayes classifier.
        
        Algorithm:
        Step 1: Calculate prior probabilities P(y)
        Step 2: Calculate feature statistics (mean, variance) per class
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
            
        Returns:
        --------
        self : fitted model
        \"\"\"
        n_samples, n_features = X.shape
        
        # Step 1: Calculate priors P(y)
        self.classes = np.unique(y)
        self.priors = {}
        
        # Step 2: Calculate feature statistics per class
        self.means = {}
        self.variances = {}
        
        print(f\"Training Naive Bayes on {n_samples} samples with {n_features} features...\")
        
        for cls in self.classes:
            X_cls = X[y == cls]
            
            # Prior probability
            self.priors[cls] = len(X_cls) / n_samples
            
            # Feature statistics (mean and variance for Gaussian assumption)
            self.means[cls] = X_cls.mean(axis=0)
            self.variances[cls] = X_cls.var(axis=0)
        
        print(f\"✓ Training complete!\")
        print(f\"Classes: {self.classes}\")
        print(f\"Prior probabilities: {self.priors}\")
        
        return self
    
    def _calculate_gaussian_pdf(self, x, mean, variance):
        \"\"\"
        Calculate Gaussian PDF: P(x|μ, σ²)
        
        Formula: (1 / sqrt(2π σ²)) * exp(-(x-μ)² / 2σ²)
        
        Parameters:
        -----------
        x : value
        mean : μ
        variance : σ²
        
        Returns:
        --------
        probability : P(x|μ, σ²)
        \"\"\"
        numerator = np.exp(-(x - mean) ** 2 / (2 * (variance + self.eps)))
        denominator = np.sqrt(2 * np.pi * (variance + self.eps))
        return numerator / denominator
    
    def _calculate_posterior(self, x):
        \"\"\"
        Calculate posterior probabilities for each class.
        
        P(y|x) ∝ P(x|y) * P(y) = P(y) * Π P(x_i|y)
        
        Use log probabilities for numerical stability:
        log P(y|x) = log P(y) + Σ log P(x_i|y)
        \"\"\"
        posteriors = {}
        
        for cls in self.classes:
            # Start with log prior
            posterior = np.log(self.priors[cls])
            
            # Add log likelihoods for each feature (Naive independence assumption)
            for i in range(len(x)):
                # Gaussian likelihood
                likelihood = self._calculate_gaussian_pdf(
                    x[i],
                    self.means[cls][i],
                    self.variances[cls][i]
                )
                
                # Use log to prevent numerical underflow
                posterior += np.log(likelihood + self.eps)
            
            posteriors[cls] = posterior
        
        return posteriors
    
    def predict(self, X):
        \"\"\"
        Predict class labels for new samples.
        
        Algorithm:
        For each sample:
            1. Calculate posterior P(y|x) for each class
            2. Select class with highest posterior
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        \"\"\"
        if self.means is None:
            raise ValueError(\"Model must be fit before predictions.\")
        
        predictions = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Select class with highest posterior probability
            prediction = max(posteriors, key=posteriors.get)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        \"\"\"
        Predict class probabilities.
        
        Returns normalized posterior probabilities for each class.
        
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Probability for each class
        \"\"\"
        if self.means is None:
            raise ValueError(\"Model must be fit before predictions.\")
        
        probabilities = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert log posteriors to probabilities (softmax)
            max_posterior = max(posteriors.values())
            posteriors = {k: np.exp(v - max_posterior) for k, v in posteriors.items()}
            
            # Normalize
            total = sum(posteriors.values())
            posteriors = {k: v / total for k, v in posteriors.items()}
            
            # Create probability array in order of classes
            proba = [posteriors.get(cls, 0) for cls in sorted(self.classes)]
            probabilities.append(proba)
        
        return np.array(probabilities)


# ============================================================================
# SAMPLE USAGE: Naive Bayes Classifier
# ============================================================================
if __name__ == \"__main__\":
    print(\"=\" * 70)
    print(\"NAIVE BAYES CLASSIFIER - SAMPLE USAGE\")
    print(\"=\" * 70)
    
    # Step 1: Load iris dataset
    print(\"\\n[Step 1] Loading iris dataset...\")
    iris = __import__('sklearn.datasets', fromlist=['load_iris']).load_iris()
    X = iris.data
    y = iris.target
    
    print(f\"Dataset shape: {X.shape}\")
    print(f\"Number of classes: {len(np.unique(y))}\")
    print(f\"Feature names: {iris.feature_names}\")
    
    # Step 2: Split data
    print(\"\\n[Step 2] Splitting data (80% train, 20% test)...\")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 3: Standardize features
    print(\"\\n[Step 3] Standardizing features...\")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Train Naive Bayes
    print(\"\\n[Step 4] Training Naive Bayes Classifier...\")
    model = NaiveBayes()
    model.fit(X_train_scaled, y_train)
    
    # Step 5: Make predictions
    print(\"\\n[Step 5] Making predictions...\")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Step 6: Calculate metrics
    print(\"\\n[Step 6] Calculating metrics...\")
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f\"Training Accuracy: {train_acc:.4f} ({int(train_acc*100)}%)\")
    print(f\"Testing Accuracy:  {test_acc:.4f} ({int(test_acc*100)}%)\")
    
    # Step 7: Confusion matrix
    print(\"\\n[Step 7] Confusion Matrix (Test Set):\")
    cm = confusion_matrix(y_test, y_pred_test)
    print(\"Predicted ->\"  )
    print(cm)
    
    # Step 8: Feature means and variances
    print(\"\\n[Step 8] Model Parameters:\")
    for cls in sorted(model.classes):
        print(f\"\\nClass {cls} ({iris.target_names[cls]}):\")
        print(f\"  Prior probability: {model.priors[cls]:.4f}\")
        print(f\"  Feature means: {model.means[cls]}\")
        print(f\"  Feature vars:  {model.variances[cls]}\")
    
    # Step 9: Visualization
    print(\"\\n[Step 9] Generating visualizations...\")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy
    accuracies = [train_acc, test_acc]
    datasets = ['Training', 'Testing']
    bars = axes[0, 0].bar(datasets, accuracies, color=['skyblue', 'orange'],
                          edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Naive Bayes: Train vs Test', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Confusion matrix
    im = axes[0, 1].imshow(cm, cmap='Blues', aspect='auto')
    axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
    axes[0, 1].set_ylabel('True Label', fontsize=11)
    axes[0, 1].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(3))
    axes[0, 1].set_yticks(range(3))
    
    for i in range(3):
        for j in range(3):
            text = axes[0, 1].text(j, i, cm[i, j], ha=\"center\", va=\"center\",
                                   color=\"black\", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: Prior probabilities
    priors = [model.priors[cls] for cls in sorted(model.classes)]
    class_names = [iris.target_names[cls] for cls in sorted(model.classes)]
    axes[1, 0].bar(class_names, priors, color='steelblue', edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Probability', fontsize=11)
    axes[1, 0].set_title('Prior Probabilities', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Probability distribution on test set
    for cls in sorted(model.classes):
        mask = y_test == cls
        axes[1, 1].hist(y_proba[mask, cls], bins=15, alpha=0.6, label=iris.target_names[cls],
                       edgecolor='black', linewidth=1)
    
    axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Predicted Probabilities by Class', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('naive_bayes_visualization.png', dpi=300, bbox_inches='tight')
    print(\"✓ Visualization saved as 'naive_bayes_visualization.png'\")
    plt.show()
    
    print(\"\\n\" + \"=\" * 70)
    print(\"Naive Bayes Classifier training completed successfully!\")
    print(\"=\" * 70)
