"""
Model Training and Optimization Module
Implements XGBoost with ensemble methods, hyperparameter tuning, and calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, make_scorer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BusinessCostFunction:
    """
    Business cost function where false negatives are more costly than false positives
    Cost of missing a failure (FN) > Cost of false alarm (FP)
    """

    def __init__(self, fn_cost: float = 1000, fp_cost: float = 50):
        self.fn_cost = fn_cost  # Cost of missing a warranty claim
        self.fp_cost = fp_cost  # Cost of unnecessary inspection

    def calculate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate total business cost"""
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)
        return total_cost

    def get_scorer(self):
        """Get sklearn scorer (note: returns negative cost for maximization)"""
        def cost_scorer(y_true, y_pred):
            return -self.calculate_cost(y_true, y_pred)
        return make_scorer(cost_scorer)


class XGBoostOptimizer:
    """
    XGBoost hyperparameter optimization using Optuna
    Handles imbalanced data with SMOTE and class weights
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        seed: int = 42,
        use_smote: bool = True
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.seed = seed
        self.use_smote = use_smote
        self.best_params = None
        self.study = None

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function for XGBoost"""

        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': self.seed,
            'n_jobs': -1,
            'tree_method': 'hist',
            'enable_categorical': False
        }

        # Calculate scale_pos_weight for imbalanced data
        n_negative = (y == 'No').sum()
        n_positive = (y == 'Yes').sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        params['scale_pos_weight'] = scale_pos_weight

        # Encode target
        y_encoded = (y == 'Yes').astype(int)

        # Create model
        model = xgb.XGBClassifier(**params)

        # Apply SMOTE if enabled
        if self.use_smote:
            smote = SMOTE(random_state=self.seed, k_neighbors=5)
            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', model)
            ])
        else:
            pipeline = model

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        scores = cross_val_score(
            pipeline, X, y_encoded,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        return scores.mean()

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run Optuna optimization"""
        logger.info(f"[{datetime.now()}] Starting hyperparameter optimization with {self.n_trials} trials...")

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed)
        )

        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=False
        )

        self.best_params = self.study.best_params
        logger.info(f"[{datetime.now()}] Optimization complete. Best F1: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params


class EnsembleModel:
    """
    Ensemble of XGBoost, Random Forest, and LightGBM
    Uses soft voting for final predictions
    """

    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        use_smote: bool = True
    ):
        self.xgb_params = xgb_params or {}
        self.seed = seed
        self.use_smote = use_smote
        self.ensemble = None
        self.smote = None

    def build_ensemble(self, scale_pos_weight: float = 1.0) -> VotingClassifier:
        """Build voting classifier ensemble"""

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            tree_method='hist'
        )

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.seed,
            n_jobs=-1
        )

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced',
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1
        )

        # Soft voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lgb', lgb_model)
            ],
            voting='soft',
            weights=[2, 1, 1]  # XGBoost gets higher weight
        )

        return ensemble

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit ensemble model with SMOTE"""
        logger.info(f"[{datetime.now()}] Training ensemble model...")

        # Encode target
        y_encoded = (y == 'Yes').astype(int)

        # Calculate scale_pos_weight
        n_negative = (y == 'No').sum()
        n_positive = (y == 'Yes').sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        # Build ensemble
        self.ensemble = self.build_ensemble(scale_pos_weight)

        # Apply SMOTE
        if self.use_smote:
            logger.info("Applying SMOTE for balancing...")
            self.smote = SMOTE(random_state=self.seed, k_neighbors=5)
            X_resampled, y_resampled = self.smote.fit_resample(X, y_encoded)
        else:
            X_resampled, y_resampled = X, y_encoded

        # Train ensemble
        self.ensemble.fit(X_resampled, y_resampled)

        logger.info(f"[{datetime.now()}] Ensemble training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.ensemble.predict_proba(X)


class CalibratedModel:
    """
    Calibrated classifier using Platt scaling
    Improves probability estimates
    """

    def __init__(self, base_model, method: str = 'sigmoid', cv: int = 5):
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.calibrated_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit calibrated model"""
        logger.info(f"[{datetime.now()}] Calibrating model with {self.method} method...")

        y_encoded = (y == 'Yes').astype(int)

        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=self.method,
            cv=self.cv
        )

        self.calibrated_model.fit(X, y_encoded)

        logger.info(f"[{datetime.now()}] Calibration complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.calibrated_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities"""
        return self.calibrated_model.predict_proba(X)


class FeatureSelector:
    """
    Recursive Feature Elimination using cross-validation
    Uses feature importance for selection
    """

    def __init__(self, estimator, min_features: int = 10, cv: int = 5):
        self.estimator = estimator
        self.min_features = min_features
        self.cv = cv
        self.selector = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Perform feature selection"""
        logger.info(f"[{datetime.now()}] Starting feature selection with RFECV...")

        y_encoded = (y == 'Yes').astype(int)

        self.selector = RFECV(
            estimator=self.estimator,
            step=1,
            cv=StratifiedKFold(self.cv),
            scoring='f1',
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        self.selector.fit(X, y_encoded)
        self.selected_features = X.columns[self.selector.support_].tolist()

        logger.info(f"[{datetime.now()}] Selected {len(self.selected_features)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class USGFailurePredictionModel:
    """
    Complete USG failure prediction model
    Orchestrates optimization, ensemble, calibration, and feature selection
    """

    def __init__(
        self,
        optimize_hyperparams: bool = True,
        n_trials: int = 50,
        use_ensemble: bool = True,
        use_calibration: bool = True,
        use_feature_selection: bool = False,
        seed: int = 42
    ):
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.use_ensemble = use_ensemble
        self.use_calibration = use_calibration
        self.use_feature_selection = use_feature_selection
        self.seed = seed

        self.optimizer = None
        self.model = None
        self.feature_selector = None
        self.best_params = None
        self.training_time = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator (required by sklearn).

        Args:
            deep: If True, return parameters for nested objects

        Returns:
            Dictionary of parameter names mapped to their values
        """
        return {
            'optimize_hyperparams': self.optimize_hyperparams,
            'n_trials': self.n_trials,
            'use_ensemble': self.use_ensemble,
            'use_calibration': self.use_calibration,
            'use_feature_selection': self.use_feature_selection,
            'seed': self.seed
        }

    def set_params(self, **params):
        """
        Set parameters for this estimator (required by sklearn).

        Args:
            **params: Estimator parameters

        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the complete model"""
        start_time = datetime.now()
        logger.info(f"[{start_time}] Starting model training pipeline...")

        # 1. Hyperparameter optimization
        if self.optimize_hyperparams:
            self.optimizer = XGBoostOptimizer(
                n_trials=self.n_trials,
                seed=self.seed,
                use_smote=True
            )
            self.best_params = self.optimizer.optimize(X, y)
        else:
            # Default parameters
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0
            }

        # 2. Build and train model
        if self.use_ensemble:
            ensemble = EnsembleModel(
                xgb_params=self.best_params,
                seed=self.seed,
                use_smote=True
            )
            ensemble.fit(X, y)
            base_model = ensemble.ensemble
        else:
            # Single XGBoost model
            y_encoded = (y == 'Yes').astype(int)
            n_negative = (y == 'No').sum()
            n_positive = (y == 'Yes').sum()
            scale_pos_weight = n_negative / n_positive

            model = xgb.XGBClassifier(
                **self.best_params,
                scale_pos_weight=scale_pos_weight,
                random_state=self.seed,
                n_jobs=-1
            )

            if True:  # Always use SMOTE
                smote = SMOTE(random_state=self.seed)
                X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
                model.fit(X_resampled, y_resampled)
            else:
                model.fit(X, y_encoded)

            base_model = model

        # 3. Calibration
        if self.use_calibration:
            calibrated = CalibratedModel(base_model, method='sigmoid', cv=5)
            calibrated.fit(X, y)
            self.model = calibrated
        else:
            self.model = base_model

        end_time = datetime.now()
        self.training_time = (end_time - start_time).total_seconds()

        logger.info(f"[{end_time}] Model training complete in {self.training_time:.2f} seconds")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.use_calibration:
            predictions = self.model.predict(X)
            return np.array(['Yes' if p == 1 else 'No' for p in predictions])
        else:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X)
                return np.array(['Yes' if p == 1 else 'No' for p in predictions])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """Save model to disk"""
        logger.info(f"Saving model to {filepath}")
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'USGFailurePredictionModel':
        """Load model from disk"""
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    logger.info("USG Failure Prediction Model initialized")
    logger.info("Use USGFailurePredictionModel class for training")
