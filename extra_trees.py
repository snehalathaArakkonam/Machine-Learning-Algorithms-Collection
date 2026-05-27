"""
Extra Trees (Extremely Randomized Trees) Classifier
===================================================
Extra Trees is an ensemble method that combines multiple randomized decision trees
using bootstrap samples and random feature/threshold selection.

Key Differences from Random Forest:
1. Random threshold selection for splits (instead of searching for optimal)
2. Uses whole dataset for tree training (instead of bootstrap by default)
3. Faster to train due to random thresholds
4. Lower variance but potentially higher bias

Algorithm:
1. Create multiple bootstrap samples
2. For each tree:
    - At each node, randomly select features and thresholds
    - Choose the split that maximizes information gain
    - Grow tree to full depth
3. Aggregate predictions via majority voting
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


class Node:
    \"\"\"Decision tree node for Extra Trees.\"\"\"
    
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, samples=None):
        \"\"\"Initialize tree node.\"\"\"
        self.feature = feature          # Feature index to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Class value if leaf node
        self.samples = samples         # Number of samples at this node


class ExtraTree:
    \"\"\"Single Extremely Randomized Tree.\"\"\"
    
    def __init__(self, max_depth=10, min_samples_split=2):
        \"\"\"Initialize tree parameters.\"\"\"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.n_features = None
        self.n_classes = None
    
    def fit(self, X, y):
        \"\"\"Build tree with random feature/threshold selection.\"\"\"
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        \"\"\"
        Recursively build tree using random splits.
        
        Algorithm:
        Step 1: Check stopping criteria
        Step 2: Randomly select features and thresholds
        Step 3: Find best split
        Step 4: Recursively build left and right subtrees
        \"\"\"
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Step 1: Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Create leaf node
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common, samples=n_samples)
        
        # Step 2: Random feature/threshold selection
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        
        # Try random splits
        n_random_features = int(np.sqrt(self.n_features))
        random_features = np.random.choice(self.n_features, n_random_features, replace=False)
        
        for feature in random_features:
            # Random threshold selection
            feature_values = X[:, feature]
            min_val, max_val = feature_values.min(), feature_values.max()
            threshold = np.random.uniform(min_val, max_val)
            
            # Split data
            left_idx = feature_values <= threshold
            right_idx = ~left_idx
            
            if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                continue
            
            # Information gain
            parent_entropy = self._entropy(y)
            left_entropy = self._entropy(y[left_idx])
            right_entropy = self._entropy(y[right_idx])
            
            weighted_entropy = (np.sum(left_idx) * left_entropy +
                              np.sum(right_idx) * right_entropy) / n_samples
            gain = parent_entropy - weighted_entropy
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left_idx = left_idx
                best_right_idx = right_idx
        
        # Step 3: If no good split found
        if best_feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common, samples=n_samples)
        
        # Step 4: Recursively build subtrees
        left_child = self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1)
        right_child = self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples
        )
    
    def predict(self, X):
        \"\"\"Predict class labels.\"\"\"
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        \"\"\"Traverse tree to make prediction.\"\"\"
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def _entropy(self, y):
        \"\"\"Calculate entropy.\"\"\"
        counter = Counter(y)
        entropy = 0
        for count in counter.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy


class ExtraTreesClassifier:
    \"\"\"
    Extremely Randomized Trees Classifier.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees
    max_depth : int, default=10
        Maximum tree depth
    \"\"\"
    
    def __init__(self, n_estimators=100, max_depth=10):
        \"\"\"Initialize Extra Trees parameters.\"\"\"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        \"\"\"
        Train Extra Trees ensemble.
        
        Algorithm:
        For each tree:
            - Generate bootstrap sample
            - Build randomized tree
            - Store trained tree
        \"\"\"
        n_samples = X.shape[0]
        
        print(f\"Training Extra Trees with {self.n_estimators} trees...\")
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train tree
            tree = ExtraTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f\"  Trained {i + 1} trees\")
        
        print(f\"✓ Training complete!\")
        
        return self
    
    def predict(self, X):
        \"\"\"
        Predict class labels using majority voting.
        
        Algorithm:
        For each sample:
            - Get prediction from each tree
            - Vote: select most common prediction
        \"\"\"
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        final_predictions = []
        for j in range(X.shape[0]):
            votes = predictions[:, j]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        \"\"\"
        Predict class probabilities.
        
        Returns class probabilities based on voting proportions.
        \"\"\"
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        classes = np.unique(predictions)
        proba = np.zeros((X.shape[0], len(classes)))
        
        for j in range(X.shape[0]):
            votes = predictions[:, j]
            for i, cls in enumerate(classes):
                proba[j, i] = np.sum(votes == cls) / len(votes)
        
        return proba


# ============================================================================
# SAMPLE USAGE: Extra Trees Classifier
# ============================================================================
if __name__ == \"__main__\":
    print(\"=\" * 70)
    print(\"EXTRA TREES (EXTREMELY RANDOMIZED TREES) - SAMPLE USAGE\")
    print(\"=\" * 70)
    
    # Step 1: Load iris dataset
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
    
    # Step 4: Train Extra Trees
    print(\"\\n[Step 4] Training Extra Trees Classifier...\")
    model = ExtraTreesClassifier(n_estimators=100, max_depth=10)
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
    axes[0].set_title('Extra Trees: Train vs Test', fontsize=12, fontweight='bold')
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
    plt.savefig('extra_trees_visualization.png', dpi=300, bbox_inches='tight')
    print(\"✓ Visualization saved as 'extra_trees_visualization.png'\")
    plt.show()
    
    print(\"\\n\" + \"=\" * 70)
    print(\"Extra Trees Classifier training completed successfully!\")
    print(\"=\" * 70)
