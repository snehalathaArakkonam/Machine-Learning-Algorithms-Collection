"""
XGBoost Classifier Implementation
=================================
XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library.
It uses regularized boosting and tree pruning to improve performance and prevent overfitting.

Key Differences from Standard Gradient Boosting:
1. Regularization (L1, L2) on both weights and tree complexity
2. Column subsampling and row subsampling
3. Weighted quantile sketch for tree building
4. Out-of-core computation for large datasets
5. Parallel and distributed training

This is a wrapper using the official xgboost library for production-ready performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Try to import xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


class XGBoostClassifier:
    \"\"\"
    XGBoost Classifier Wrapper.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.1
        Step size shrinkage (eta)
    max_depth : int, default=6
        Maximum tree depth
    subsample : float, default=0.8
        Fraction of samples for tree building
    colsample_bytree : float, default=0.8
        Fraction of features for tree building
    \"\"\"
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8):
        \"\"\"Initialize XGBoost parameters.\"\"\"
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = None
    
    def fit(self, X, y):
        \"\"\"
        Train XGBoost classifier.
        
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
        params = {
            'objective': 'binary:logistic',
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'eval_metric': 'logloss'
        }
        
        dtrain = xgb.DMatrix(X, label=y)
        
        print(f"Training XGBoost with {self.n_estimators} boosting rounds...")
        print(f"Parameters: max_depth={self.max_depth}, learning_rate={self.learning_rate}")
        
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=20
        )
        
        print(f"✓ Training complete!")
        
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
            raise ValueError("Model must be fit before predictions.")
        
        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
        return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X):
        \"\"\"
        Predict class probabilities.
        
        Returns:
        --------
        probabilities : array, shape (n_samples, 2)
            Probabilities for each class
        \"\"\"
        if self.model is None:
            raise ValueError("Model must be fit before predictions.")
        
        dtest = xgb.DMatrix(X)
        proba_class1 = self.model.predict(dtest)
        proba_class0 = 1 - proba_class1
        
        return np.column_stack([proba_class0, proba_class1])
    
    def get_feature_importance(self):
        \"\"\"Get feature importance scores.\"\"\"
        if self.model is None:
            raise ValueError("Model must be fit before getting importance.")
        
        return self.model.get_score(importance_type='weight')


# ============================================================================
# SAMPLE USAGE: XGBoost Classifier
# ============================================================================
if __name__ == \"__main__\":
    if not XGBOOST_AVAILABLE:
        print(\"XGBoost not available. Install with: pip install xgboost\")
    else:
        print(\"=\" * 70)
        print(\"XGBOOST CLASSIFIER - SAMPLE USAGE\")
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
        
        # Step 4: Train XGBoost
        print(\"\\n[Step 4] Training XGBoost Classifier...\")
        model = XGBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
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
        
        # Step 8: Visualization
        print(\"\\n[Step 8] Generating visualizations...\")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy
        accuracies = [train_acc, test_acc]
        datasets = ['Training', 'Testing']
        bars = axes[0].bar(datasets, accuracies, color=['skyblue', 'orange'],
                          edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].set_title('XGBoost: Train vs Test', fontsize=12, fontweight='bold')
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
        plt.savefig('xgboost_visualization.png', dpi=300, bbox_inches='tight')
        print(\"✓ Visualization saved as 'xgboost_visualization.png'\")
        plt.show()
        
        print(\"\\n\" + \"=\" * 70)
        print(\"XGBoost Classifier training completed successfully!\")
        print(\"=\" * 70)
