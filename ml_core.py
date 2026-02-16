"""
USG Failure Prediction - Unified ML Pipeline
Single-file production module for training and saving the model.

Run directly from project root:
    python ml_core.py

Outputs:
    - models/model.pkl
    - models/preprocessor.pkl
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, classification_report, confusion_matrix
)

import xgboost as xgb

warnings.filterwarnings('ignore')


class USGModelPipeline:
    """
    Unified pipeline for USG failure prediction.
    Handles data loading, preprocessing, training, and model export.
    """

    # Data leakage columns to drop
    # Manufacturing Focus: Drop non-technical features that cause noise
    # - Customer_Region: Geographic patterns mask true manufacturing root causes
    # - Installation_Date: Temporal patterns distract from technical defects
    LEAKAGE_COLUMNS = [
        'Device_UUID', 'Serial_Number', 'Claim_Type', 'Repair_Cost_USD',
        'Customer_Region',  # Non-technical: geographic noise
        'Installation_Date'  # Non-technical: temporal noise
    ]

    # XGBoost hyperparameters (optimized for production)
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }

    def __init__(self, data_path='data/raw/USG_Data_cleared.csv', random_state=42):
        """
        Initialize the pipeline.

        Args:
            data_path: Path to the raw CSV file (relative to project root)
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'Warranty_Claim'

        # Create output directories
        Path('models').mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)

        print("="*70)
        print("USG FAILURE PREDICTION - UNIFIED ML PIPELINE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

    def load_data(self):
        """Load and validate the dataset."""
        print(f"\n[1/5] Loading data from: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"\n❌ Data file not found: {self.data_path}\n"
                f"Please ensure USG_Data_cleared.csv is in the data/raw/ directory."
            )

        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Validate target column
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Display target distribution
        target_counts = df[self.target_column].value_counts()
        failure_rate = (df[self.target_column] == 'Yes').mean() * 100

        print(f"\nTarget distribution:")
        for label, count in target_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label:5s}: {count:4d} ({pct:5.2f}%)")
        print(f"  Failure rate: {failure_rate:.2f}%")

        return df

    def engineer_features(self, X):
        """
        Create manufacturing-focused interaction features.

        Critical for root cause analysis:
        - Solder_Temperature_x_Supplier: Identifies if specific suppliers'
          components fail under certain thermal conditions

        This forces the model to learn technical manufacturing relationships
        rather than relying on geographic or temporal patterns.

        Args:
            X: Feature dataframe

        Returns:
            X with added interaction features
        """
        print(f"\n[2/6] Engineering manufacturing interaction features")

        X = X.copy()
        features_created = 0

        # Critical interaction: Solder Temperature × Supplier
        if 'Solder_Temperature' in X.columns and 'Component_Supplier' in X.columns:
            X['Solder_Temperature_x_Supplier'] = (
                X['Solder_Temperature'].astype(str) + '_' +
                X['Component_Supplier'].astype(str)
            )
            features_created += 1
            print(f"✓ Created: Solder_Temperature_x_Supplier")
            print(f"  → {X['Solder_Temperature_x_Supplier'].nunique()} unique combinations")

        # Optional: Soldering Time × Humidity (thermal stress indicator)
        if 'Soldering_Time' in X.columns and 'Ambient_Humidity' in X.columns:
            X['Soldering_Time_x_Humidity'] = (
                X['Soldering_Time'].astype(str) + '_' +
                X['Ambient_Humidity'].astype(str)
            )
            features_created += 1
            print(f"✓ Created: Soldering_Time_x_Humidity")

        print(f"✓ Total interaction features created: {features_created}")

        return X

    def build_preprocessor(self, X):
        """
        Build preprocessing pipeline.

        Steps:
        1. Drop leakage columns
        2. Identify numeric and categorical features
        3. Build transformation pipeline (impute → scale/encode)
        """
        print(f"\n[3/6] Building preprocessing pipeline")

        # Drop leakage columns
        leakage_cols_present = [col for col in self.LEAKAGE_COLUMNS if col in X.columns]
        if leakage_cols_present:
            print(f"✓ Dropping {len(leakage_cols_present)} leakage columns: {leakage_cols_present}")
            X = X.drop(columns=leakage_cols_present)

        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        print(f"✓ Numeric features: {len(numeric_features)}")
        print(f"✓ Categorical features: {len(categorical_features)}")

        # Build numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Build categorical transformer (Label Encoding for tree models)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('label', LabelEncoderPipeline())  # Custom label encoder
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )

        self.feature_columns = numeric_features + categorical_features
        print(f"✓ Pipeline built: {len(self.feature_columns)} total features")

        return preprocessor, X

    def train_model(self, X, y):
        """
        Train XGBoost model with fixed hyperparameters.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Trained model
        """
        print(f"\n[4/6] Training XGBoost model")

        # Encode target (Yes=1, No=0)
        y_encoded = (y == 'Yes').astype(int)

        # Calculate scale_pos_weight for class imbalance
        n_negative = (y_encoded == 0).sum()
        n_positive = (y_encoded == 1).sum()
        scale_pos_weight = n_negative / n_positive

        print(f"✓ Class distribution: {n_negative} negative, {n_positive} positive")
        print(f"✓ scale_pos_weight: {scale_pos_weight:.2f}")

        # Fit preprocessor
        print("✓ Fitting preprocessor...")
        self.preprocessor.fit(X)

        # Transform data
        X_transformed = self.preprocessor.transform(X)
        print(f"✓ Data transformed: {X_transformed.shape}")

        # Configure XGBoost with scale_pos_weight
        params = self.XGB_PARAMS.copy()
        params['scale_pos_weight'] = scale_pos_weight

        print(f"✓ XGBoost hyperparameters:")
        for key, value in params.items():
            if key not in ['random_state', 'n_jobs', 'tree_method', 'eval_metric']:
                print(f"    {key:20s} = {value}")

        # Train model
        print("✓ Training model...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_transformed, y_encoded)

        print(f"✓ Model trained successfully")

        return self.model

    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Feature dataframe
            y: Target series
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        print(f"\n[5/6] Cross-validation ({cv_folds}-fold)")

        # Encode target
        y_encoded = (y == 'Yes').astype(int)

        # Transform data
        X_transformed = self.preprocessor.transform(X)

        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Scoring metrics
        scoring = {
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc'
        }

        results = {}
        for metric_name, metric in scoring.items():
            scores = cross_val_score(
                self.model, X_transformed, y_encoded,
                cv=cv, scoring=metric, n_jobs=-1
            )
            results[f'{metric_name}_mean'] = scores.mean()
            results[f'{metric_name}_std'] = scores.std()

            print(f"  {metric_name:12s}: {scores.mean():.4f} (±{scores.std():.4f})")

        return results

    def evaluate(self, X, y):
        """
        Final evaluation on full dataset (for reference).

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n[6/6] Final Evaluation")

        # Encode target
        y_encoded = (y == 'Yes').astype(int)

        # Transform and predict
        X_transformed = self.preprocessor.transform(X)
        y_pred = self.model.predict(X_transformed)
        y_proba = self.model.predict_proba(X_transformed)[:, 1]

        # Calculate metrics
        metrics = {
            'f1_score': f1_score(y_encoded, y_pred),
            'precision': precision_score(y_encoded, y_pred),
            'recall': recall_score(y_encoded, y_pred),
            'accuracy': accuracy_score(y_encoded, y_pred),
            'roc_auc': roc_auc_score(y_encoded, y_proba)
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_encoded, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })

        # Display results
        print(f"\nPerformance Metrics:")
        print(f"  F1 Score:    {metrics['f1_score']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  |  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  |  TN: {tn:4d}")

        return metrics

    def save_models(self):
        """Save trained model and preprocessor to disk."""
        print(f"\nSaving models...")

        # Save model
        model_path = Path('models/model.pkl')
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved: {model_path}")

        # Save preprocessor
        preprocessor_path = Path('models/preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✓ Preprocessor saved: {preprocessor_path}")

        # Save feature names for reference
        feature_names = {
            'features': self.feature_columns,
            'n_features': len(self.feature_columns)
        }
        import json
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ Feature names saved: models/feature_names.json")

    def run(self):
        """Execute the complete pipeline."""
        try:
            start_time = datetime.now()

            # Step 1: Load data
            df = self.load_data()

            # Step 2: Separate features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            # Step 2.5: Engineer manufacturing interaction features
            X_engineered = self.engineer_features(X)

            # Step 3: Build preprocessor
            self.preprocessor, X_clean = self.build_preprocessor(X_engineered)

            # Step 4: Train model
            self.train_model(X_clean, y)

            # Step 5: Cross-validation
            cv_results = self.cross_validate(X_clean, y)

            # Step 6: Evaluate
            eval_metrics = self.evaluate(X_clean, y)

            # Step 7: Save models
            self.save_models()

            # Final summary
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            print("\n" + "="*70)
            print("PIPELINE COMPLETE")
            print("="*70)
            print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
            print(f"\nModel artifacts saved to:")
            print(f"  - models/model.pkl")
            print(f"  - models/preprocessor.pkl")
            print(f"  - models/feature_names.json")
            print(f"\nNext steps:")
            print(f"  1. Run SHAP analysis: python scripts/run_shap_analysis.py")
            print(f"  2. Start API: uvicorn src.api:app --reload")
            print("="*70)

            return {
                'cv_results': cv_results,
                'eval_metrics': eval_metrics,
                'elapsed_time': elapsed
            }

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class LabelEncoderPipeline(BaseEstimator, TransformerMixin):
    """
    Custom label encoder that works in sklearn pipelines.
    Handles unseen categories gracefully.
    """

    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        """Fit label encoders for each column."""
        X = pd.DataFrame(X)
        self.encoders_ = {}  # Add underscore to indicate fitted attribute
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        self.n_features_in_ = X.shape[1]  # Required by sklearn
        return self

    def transform(self, X):
        """Transform using fitted encoders."""
        X = pd.DataFrame(X)
        X_encoded = X.copy()

        for col in X.columns:
            if col in self.encoders_:
                le = self.encoders_[col]
                # Handle unseen categories
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        return X_encoded.values

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    # Execute the pipeline
    pipeline = USGModelPipeline(
        data_path='data/raw/USG_Data_cleared.csv',
        random_state=42
    )

    results = pipeline.run()
