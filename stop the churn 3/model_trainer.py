import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_score, recall_score
import shap
import joblib
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self):
        # Initialize individual models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = None
        self.explainer = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess the data with scaling and SMOTE if training."""
        if training:
            X_scaled = self.scaler.fit_transform(X)
            X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
            return X_resampled, y_resampled
        else:
            return self.scaler.transform(X)
    
    def train(self, X, y):
        """Train the ensemble model and create SHAP explainer."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Preprocess training data
        X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train)
        X_test_processed = self.preprocess_data(X_test, training=False)
        
        # Train individual models
        self.rf_model.fit(X_train_processed, y_train_processed)
        self.lgb_model.fit(X_train_processed, y_train_processed)
        self.xgb_model.fit(X_train_processed, y_train_processed)
        
        # Create SHAP explainer using Random Forest (most interpretable)
        self.explainer = shap.TreeExplainer(self.rf_model)
        
        # Get predictions from each model
        rf_pred = self.rf_model.predict_proba(X_test_processed)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_test_processed)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X_test_processed)[:, 1]
        
        # Ensemble predictions (simple average)
        ensemble_pred = (rf_pred + lgb_pred + xgb_pred) / 3
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred > 0.5)
        precision = precision_score(y_test, ensemble_pred > 0.5)
        recall = recall_score(y_test, ensemble_pred > 0.5)
        
        return {
            'auc_score': auc_score,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_report(y_test, ensemble_pred > 0.5)
        }
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        X_processed = self.preprocess_data(X, training=False)
        
        # Get predictions from each model
        rf_pred = self.rf_model.predict_proba(X_processed)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X_processed)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X_processed)[:, 1]
        
        # Ensemble predictions
        ensemble_pred = (rf_pred + lgb_pred + xgb_pred) / 3
        
        return ensemble_pred
    
    def get_risk_category(self, probability):
        """Categorize customers into risk levels based on churn probability."""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_feature_importance(self):
        """Get feature importance scores from all models."""
        # Get feature importance from each model
        rf_importance = self.rf_model.feature_importances_
        lgb_importance = self.lgb_model.feature_importances_
        xgb_importance = self.xgb_model.feature_importances_
        
        # Average the importance scores
        avg_importance = (rf_importance + lgb_importance + xgb_importance) / 3
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def get_shap_values(self, X):
        """Get SHAP values for feature importance explanation."""
        if self.explainer is None:
            raise ValueError("Model must be trained before getting SHAP values")
        X_processed = self.preprocess_data(X, training=False)
        return self.explainer.shap_values(X_processed)
    
    def save_model(self, path):
        """Save the trained models."""
        models = {
            'rf_model': self.rf_model,
            'lgb_model': self.lgb_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(models, path)
    
    def load_model(self, path):
        """Load trained models."""
        models = joblib.load(path)
        self.rf_model = models['rf_model']
        self.lgb_model = models['lgb_model']
        self.xgb_model = models['xgb_model']
        self.scaler = models['scaler']
        self.feature_names = models['feature_names']
        self.explainer = shap.TreeExplainer(self.rf_model) 