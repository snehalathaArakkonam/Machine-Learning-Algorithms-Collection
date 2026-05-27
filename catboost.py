"""
CatBoost Classifier Implementation
==================================
CatBoost (Categorical Boosting) is a gradient boosting framework optimized for
handling categorical features without requiring manual encoding.

Key Features:
1. Native handling of categorical features
2. Ordered boosting to reduce overfitting
3. Automatic feature scaling
4. Fast inference and prediction
5. Robust to outliers and missing values

This is a wrapper using the official catboost library.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Try to import catboost
try:
    from catboost import CatBoostClassifier as CatBoostLib
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")


class CatBoostClassifier:
    \"\"\"
    CatBoost Classifier Wrapper.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.1
        Learning rate (shrinkage)
    depth : int, default=6
        Tree depth
    subsample : float, default=0.8
        Row subsampling ratio
    verbose : bool, default=False
        Verbose output
    \"\"\"
    
    def __init__(self, n_estimators=100, learning_rate=0.1, depth=6,
                 subsample=0.8, verbose=False):
        \"\"\"Initialize CatBoost parameters.\"\"\"
        if not CATBOOST_AVAILABLE:
            raise ImportError(\"CatBoost is not installed. Install with: pip install catboost\")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.subsample = subsample
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        \"\"\"
        Train CatBoost classifier.
        
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
        print(f\"Training CatBoost with {self.n_estimators} iterations...\")
        print(f\"Parameters: depth={self.depth}, learning_rate={self.learning_rate}\")
        
        self.model = CatBoostLib(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.depth,
            subsample=self.subsample,
            verbose=100 if self.verbose else 0,
            random_state=42
        )
        
        self.model.fit(X, y, verbose=False)
        
        print(f\"✓ Training complete!\")
        
        return self
    
    def predict(self, X):
        \"\"\"
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        \"\"\"
        if self.model is None:
            raise ValueError(\"Model must be fit before predictions.\")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        \"\"\"
        Predict class probabilities.
        
        Returns:
        --------
        probabilities : array, shape (n_samples, 2)
            Probabilities for each class
        \"\"\"
        if self.model is None:
            raise ValueError(\"Model must be fit before predictions.\")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        \"\"\"Get feature importance scores.\"\"\"
        if self.model is None:
            raise ValueError(\"Model must be fit before getting importance.\")
        
        return self.model.get_feature_importance()


# ============================================================================
# SAMPLE USAGE: CatBoost Classifier
# ============================================================================
if __name__ == \"__main__\":
    if not CATBOOST_AVAILABLE:
        print(\"CatBoost not available. Install with: pip install catboost\")
    else:
        print(\"=\" * 70)
        print(\"CATBOOST CLASSIFIER - SAMPLE USAGE\")
        print(\"=\" * 70)
        
        # Step 1: Load dataset
        print(\"\\n[Step 1] Loading breast cancer dataset...\")
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        print(f\"Dataset shape: {X.shape}\")
        print(f\"Feature names: {len(data.feature_names)} features\")
        
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
        
        # Step 4: Train CatBoost
        print(\"\\n[Step 4] Training CatBoost Classifier...\")
        model = CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=6)
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
        print(f\"               Predicted 0  Predicted 1\")
        print(f\"Actual 0:      {cm[0, 0]:>11}  {cm[0, 1]:>11}\")
        print(f\"Actual 1:      {cm[1, 0]:>11}  {cm[1, 1]:>11}\")
        
        # Step 8: Feature importance
        print(\"\\n[Step 8] Top 10 Important Features:\")
        importance = model.get_feature_importance()
        top_indices = np.argsort(importance)[-10:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f\"  {rank:2}. {data.feature_names[idx]:30} {importance[idx]:8.2f}\")
        
        # Step 9: Visualization
        print(\"\\n[Step 9] Generating visualizations...\")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy
        accuracies = [train_acc, test_acc]
        datasets = ['Training', 'Testing']
        bars = axes[0].bar(datasets, accuracies, color=['skyblue', 'orange'],
                          edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_title('CatBoost: Train vs Test', fontsize=12, fontweight='bold')
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
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        
        for i in range(2):
            for j in range(2):
                text = axes[1].text(j, i, cm[i, j], ha=\"center\", va=\"center\",
                                   color=\"black\", fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('catboost_visualization.png', dpi=300, bbox_inches='tight')
        print(\"✓ Visualization saved as 'catboost_visualization.png'\")
        plt.show()
        
        print(\"\\n\" + \"=\" * 70)
        print(\"CatBoost Classifier training completed successfully!\")
        print(\"=\" * 70)
