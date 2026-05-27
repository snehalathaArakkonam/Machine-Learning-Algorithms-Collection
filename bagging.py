"""
Bagging (Bootstrap Aggregating) Classifier Implementation
========================================================
Bagging is an ensemble method that combines multiple independent base learners
trained on bootstrap samples of the same dataset.

Algorithm:
1. Create multiple bootstrap samples (random sampling with replacement)
2. Train a base learner on each bootstrap sample
3. For classification: aggregate predictions via majority voting
4. For regression: aggregate predictions via averaging

Key Benefits:
- Reduces variance by averaging independent learners
- Useful for high-variance models (decision trees)
- Can be parallelized since base learners are independent
- Works well with unstable learners

This implementation uses decision trees as base learners.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter


class BaggingClassifier:
    """
    Bagging (Bootstrap Aggregating) Classifier.
    
    Parameters:
    -----------
    n_estimators : int, default=50
        Number of base learners
    max_samples : int or float, default=1.0
        Number of samples for each bootstrap
    max_features : int or float, default=1.0
        Number of features for each learner
    bootstrap : bool, default=True
        Whether to use bootstrap sampling
    random_state : int, default=None
        Random seed
    """
    
    def __init__(self, n_estimators=50, max_samples=1.0, max_features=1.0,
                 bootstrap=True, random_state=None):
        """Initialize Bagging parameters."""
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.base_learners = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y):
        """
        Train Bagging ensemble.
        
        Algorithm:
        Step 1: For each base learner:
            a. Create bootstrap sample (sample with replacement)
            b. Train base learner on bootstrap sample
            c. Store trained base learner
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
            
        Returns:
        --------
        self : fitted model
        """
        n_samples, n_features = X.shape
        
        # Determine number of samples and features per bootstrap
        n_samples_per_bag = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else self.max_samples
        n_features_per_bag = int(self.max_features * n_features) if isinstance(self.max_features, float) else self.max_features
        
        print(f\"Training Bagging ensemble with {self.n_estimators} base learners...\")
        
        # Step 1: Train base learners
        for i in range(self.n_estimators):
            # Bootstrap sampling (sample with replacement)
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples_per_bag, replace=True)
            else:
                indices = np.random.choice(n_samples, n_samples_per_bag, replace=False)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Feature subsampling
            feature_indices = np.random.choice(n_features, n_features_per_bag, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]
            
            # Train decision tree
            base_learner = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
            base_learner.fit(X_bootstrap, y_bootstrap)
            
            # Store learner with feature indices for later prediction
            self.base_learners.append({
                'learner': base_learner,
                'features': feature_indices
            })
            
            if (i + 1) % 10 == 0:
                print(f\"  Trained {i + 1} base learners\")
        
        print(f\"✓ Training complete!\")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using ensemble majority voting.
        
        Algorithm:
        For each sample:
            - Get prediction from each base learner
            - Majority voting (most common prediction)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        if len(self.base_learners) == 0:
            raise ValueError(\"Model must be fit before predictions.\")
        
        predictions = []
        
        for sample in X:
            votes = []
            
            for learner_dict in self.base_learners:
                learner = learner_dict['learner']
                features = learner_dict['features']
                
                # Get prediction from this base learner
                sample_subset = sample[features].reshape(1, -1)
                pred = learner.predict(sample_subset)[0]
                votes.append(pred)
            
            # Majority voting
            most_common = Counter(votes).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Returns class probabilities based on voting proportions.
        """
        if len(self.base_learners) == 0:
            raise ValueError(\"Model must be fit before predictions.\")
        
        proba_list = []
        
        for sample in X:
            votes = []
            
            for learner_dict in self.base_learners:
                learner = learner_dict['learner']
                features = learner_dict['features']
                
                # Get prediction from this base learner
                sample_subset = sample[features].reshape(1, -1)
                pred = learner.predict(sample_subset)[0]
                votes.append(pred)
            
            # Calculate probabilities
            vote_counts = Counter(votes)
            classes = sorted(vote_counts.keys())
            
            proba = []
            for cls in classes:
                proba.append(vote_counts[cls] / len(votes))
            
            proba_list.append(proba)
        
        return np.array(proba_list)


# ============================================================================
# SAMPLE USAGE: Bagging Classifier
# ============================================================================
if __name__ == \"__main__\":
    print(\"=\" * 70)
    print(\"BAGGING (BOOTSTRAP AGGREGATING) - SAMPLE USAGE\")
    print(\"=\" * 70)
    
    # Step 1: Load dataset
    print(\"\\n[Step 1] Loading iris dataset...\")
    iris = __import__('sklearn.datasets', fromlist=['load_iris']).load_iris()
    X = iris.data
    y = iris.target
    
    print(f\"Dataset shape: {X.shape}\")
    print(f\"Number of classes: {len(np.unique(y))}\")
    
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
    
    # Step 4: Train Bagging
    print(\"\\n[Step 4] Training Bagging Classifier...\")
    print(\"Parameters: n_estimators=50, max_samples=1.0, max_features=1.0\")
    
    model = BaggingClassifier(n_estimators=50, max_samples=1.0, max_features=1.0,
                             bootstrap=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Step 5: Make predictions
    print(\"\\n[Step 5] Making predictions...\")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Step 6: Calculate metrics
    print(\"\\n[Step 6] Calculating metrics...\")
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f\"Training Accuracy: {train_acc:.4f} ({int(train_acc*100)}%)\")
    print(f\"Testing Accuracy:  {test_acc:.4f} ({int(test_acc*100)}%)\")
    
    # Step 7: Confusion matrix
    print(\"\\n[Step 7] Confusion Matrix (Test Set):\")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Step 8: Visualization
    print(\"\\n[Step 8] Generating visualizations...\")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    accuracies = [train_acc, test_acc]
    datasets = ['Training', 'Testing']
    bars = axes[0].bar(datasets, accuracies, color=['skyblue', 'orange'],
                      edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Bagging: Train vs Test', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Confusion matrix
    im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    
    for i in range(3):
        for j in range(3):
            text = axes[1].text(j, i, cm[i, j], ha=\"center\", va=\"center\",
                               color=\"black\", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('bagging_visualization.png', dpi=300, bbox_inches='tight')
    print(\"✓ Visualization saved as 'bagging_visualization.png'\")
    plt.show()
    
    print(\"\\n\" + \"=\" * 70)
    print(\"Bagging Classifier training completed successfully!\")
    print(\"=\" * 70)
