"""
Preprocessing and Feature Engineering Pipeline
Handles data cleaning, feature creation, and transformation for USG failure prediction
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLeakageHandler:
    """Handles removal of columns that could cause data leakage"""

    LEAKAGE_COLUMNS = ['Device_UUID', 'Serial_Number', 'Claim_Type', 'Repair_Cost_USD']

    @staticmethod
    def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that could cause data leakage"""
        logger.info(f"Dropping leakage columns: {DataLeakageHandler.LEAKAGE_COLUMNS}")
        existing_cols = [col for col in DataLeakageHandler.LEAKAGE_COLUMNS if col in df.columns]
        return df.drop(columns=existing_cols, errors='ignore')


class BatchFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates batch-based features including age and quality indicators"""

    def __init__(self):
        self.batch_stats = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Calculate batch statistics from training data"""
        if 'Batch_ID' in X.columns:
            logger.info("Computing batch statistics...")

            # Extract batch number and calculate age-related features
            X_copy = X.copy()
            X_copy['Batch_Number'] = X_copy['Batch_ID'].str.extract(r'(\d+)').astype(float)

            if y is not None:
                # Calculate failure rate per batch
                batch_failure = pd.DataFrame({
                    'Batch_ID': X_copy['Batch_ID'],
                    'Warranty_Claim': y
                })
                self.batch_stats['failure_rate'] = (
                    batch_failure.groupby('Batch_ID')['Warranty_Claim']
                    .apply(lambda x: (x == 'Yes').mean() if len(x) > 0 else 0)
                    .to_dict()
                )

            # Calculate batch size
            self.batch_stats['batch_size'] = X_copy['Batch_ID'].value_counts().to_dict()

            # Calculate batch number stats for normalization
            self.batch_stats['batch_number_mean'] = X_copy['Batch_Number'].mean()
            self.batch_stats['batch_number_std'] = X_copy['Batch_Number'].std()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply batch-based feature engineering"""
        X_copy = X.copy()

        if 'Batch_ID' in X_copy.columns:
            # Extract batch number
            X_copy['Batch_Number'] = X_copy['Batch_ID'].str.extract(r'(\d+)').astype(float)

            # Batch age (normalized batch number as proxy)
            if 'batch_number_mean' in self.batch_stats:
                X_copy['Batch_Age'] = (
                    (X_copy['Batch_Number'] - self.batch_stats['batch_number_mean']) /
                    (self.batch_stats['batch_number_std'] + 1e-6)
                )

            # Batch failure rate
            if 'failure_rate' in self.batch_stats:
                X_copy['Batch_Failure_Rate'] = X_copy['Batch_ID'].map(
                    self.batch_stats['failure_rate']
                ).fillna(self.batch_stats['failure_rate'].get('global_mean', 0.0952))

            # Batch size
            if 'batch_size' in self.batch_stats:
                X_copy['Batch_Size'] = X_copy['Batch_ID'].map(
                    self.batch_stats['batch_size']
                ).fillna(X_copy['Batch_ID'].map(self.batch_stats['batch_size']).median())

        return X_copy


class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """Creates interaction features between key variables"""

    INTERACTIONS = [
        ('Assembly_Temp_C', 'Humidity_Percent'),
        ('Torque_Nm', 'Gap_mm'),
        ('Solder_Temp_C', 'Solder_Time_s'),
        ('Assembly_Temp_C', 'Solder_Temp_C'),
    ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        X_copy = X.copy()

        logger.info("Creating interaction features...")
        for feat1, feat2 in self.INTERACTIONS:
            if feat1 in X_copy.columns and feat2 in X_copy.columns:
                # Multiplicative interaction
                X_copy[f'{feat1}_x_{feat2}'] = X_copy[feat1] * X_copy[feat2]

                # Ratio (with safeguard against division by zero)
                X_copy[f'{feat1}_div_{feat2}'] = X_copy[feat1] / (X_copy[feat2] + 1e-6)

        return X_copy


class SupplierFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates supplier-based aggregate features"""

    def __init__(self):
        self.supplier_stats = {}
        self.supplier_columns = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Calculate supplier statistics from training data"""
        # Identify supplier columns
        self.supplier_columns = [col for col in X.columns if 'Supplier' in col]

        if y is not None and len(self.supplier_columns) > 0:
            logger.info(f"Computing supplier statistics for {len(self.supplier_columns)} supplier columns...")

            for col in self.supplier_columns:
                # Calculate failure rate per supplier
                supplier_failure = pd.DataFrame({
                    col: X[col],
                    'Warranty_Claim': y
                })

                self.supplier_stats[col] = (
                    supplier_failure.groupby(col)['Warranty_Claim']
                    .apply(lambda x: (x == 'Yes').mean() if len(x) > 0 else 0)
                    .to_dict()
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply supplier-based feature engineering"""
        X_copy = X.copy()

        for col in self.supplier_columns:
            if col in self.supplier_stats:
                # Map supplier to failure rate
                X_copy[f'{col}_Failure_Rate'] = X_copy[col].map(
                    self.supplier_stats[col]
                ).fillna(0.0952)  # Use global failure rate as default

        return X_copy


class AnomalyScoreGenerator(BaseEstimator, TransformerMixin):
    """Generates anomaly scores for environmental and process parameters"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = None
        self.feature_columns = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit Isolation Forest on environmental parameters"""
        # Select numeric environmental features
        environmental_features = [
            'Assembly_Temp_C', 'Humidity_Percent', 'Solder_Temp_C',
            'Solder_Time_s', 'Torque_Nm', 'Gap_mm'
        ]

        self.feature_columns = [col for col in environmental_features if col in X.columns]

        if len(self.feature_columns) > 0:
            logger.info(f"Fitting Isolation Forest on {len(self.feature_columns)} features...")
            X_numeric = X[self.feature_columns].fillna(X[self.feature_columns].median())

            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_numeric)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate anomaly scores"""
        X_copy = X.copy()

        if self.isolation_forest is not None and len(self.feature_columns) > 0:
            X_numeric = X_copy[self.feature_columns].fillna(X_copy[self.feature_columns].median())

            # Get anomaly scores (lower scores = more anomalous)
            X_copy['Anomaly_Score'] = self.isolation_forest.decision_function(X_numeric)
            X_copy['Is_Anomaly'] = (self.isolation_forest.predict(X_numeric) == -1).astype(int)

        return X_copy


class TimeSeriesFeatureCreator(BaseEstimator, TransformerMixin):
    """Creates time-series features if Serial_Number follows chronological pattern"""

    def __init__(self):
        self.has_time_pattern = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Check if Serial_Number has chronological pattern"""
        if 'Serial_Number' in X.columns:
            # Extract numeric part from serial number
            X_copy = X.copy()
            X_copy['Serial_Numeric'] = X_copy['Serial_Number'].str.extract(r'(\d+)').astype(float)

            # Check if there's a pattern (not all NaN and some variance)
            if X_copy['Serial_Numeric'].notna().sum() > len(X) * 0.5:
                self.has_time_pattern = True
                logger.info("Detected time pattern in Serial_Number")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-series features based on serial number"""
        X_copy = X.copy()

        if self.has_time_pattern and 'Serial_Number' in X_copy.columns:
            X_copy['Serial_Numeric'] = X_copy['Serial_Number'].str.extract(r'(\d+)').astype(float)

            # Normalized serial position
            min_serial = X_copy['Serial_Numeric'].min()
            max_serial = X_copy['Serial_Numeric'].max()
            X_copy['Serial_Position'] = (
                (X_copy['Serial_Numeric'] - min_serial) / (max_serial - min_serial + 1e-6)
            )

        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Handles categorical variable encoding"""

    def __init__(self):
        self.label_encoders = {}
        self.categorical_columns = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit label encoders for categorical columns"""
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Fitting encoders for {len(self.categorical_columns)} categorical columns...")
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Fit on non-null values
            non_null_values = X[col].dropna().astype(str)
            if len(non_null_values) > 0:
                le.fit(non_null_values)
                self.label_encoders[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns"""
        X_copy = X.copy()

        for col in self.categorical_columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]

                # Handle unseen categories
                X_copy[col] = X_copy[col].astype(str)
                mask = X_copy[col].isin(le.classes_)
                X_copy.loc[mask, col] = le.transform(X_copy.loc[mask, col])
                X_copy.loc[~mask, col] = -1  # Unseen category
                X_copy[col] = X_copy[col].astype(float)

        return X_copy


class USGPreprocessingPipeline:
    """
    Complete preprocessing pipeline for USG failure prediction
    Orchestrates all feature engineering and data transformation steps
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

        # Initialize transformers
        self.batch_engineer = BatchFeatureEngineer()
        self.interaction_creator = InteractionFeatureCreator()
        self.supplier_engineer = SupplierFeatureEngineer()
        self.anomaly_generator = AnomalyScoreGenerator()
        self.timeseries_creator = TimeSeriesFeatureCreator()
        self.categorical_encoder = CategoricalEncoder()

        self.imputer = None
        self.scaler = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'USGPreprocessingPipeline':
        """Fit the preprocessing pipeline"""
        logger.info(f"[{datetime.now()}] Starting preprocessing pipeline fit...")

        # 1. Drop leakage columns
        X_processed = DataLeakageHandler.drop_leakage_columns(X)

        # 2. Batch feature engineering
        self.batch_engineer.fit(X_processed, y)
        X_processed = self.batch_engineer.transform(X_processed)

        # 3. Create interaction features
        X_processed = self.interaction_creator.fit_transform(X_processed, y)

        # 4. Supplier feature engineering
        self.supplier_engineer.fit(X_processed, y)
        X_processed = self.supplier_engineer.transform(X_processed)

        # 5. Time series features
        self.timeseries_creator.fit(X_processed, y)
        X_processed = self.timeseries_creator.transform(X_processed)

        # 6. Encode categorical variables
        self.categorical_encoder.fit(X_processed, y)
        X_processed = self.categorical_encoder.transform(X_processed)

        # 7. Generate anomaly scores (must be after encoding)
        self.anomaly_generator.fit(X_processed, y)
        X_processed = self.anomaly_generator.transform(X_processed)

        # 8. Impute missing values
        self.imputer = SimpleImputer(strategy='median')
        X_processed = pd.DataFrame(
            self.imputer.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )

        # 9. Scale features
        self.scaler = StandardScaler()
        X_processed = pd.DataFrame(
            self.scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )

        self.feature_names = X_processed.columns.tolist()

        logger.info(f"[{datetime.now()}] Pipeline fit complete. Final features: {len(self.feature_names)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        logger.info(f"[{datetime.now()}] Applying preprocessing transformations...")

        # Apply all transformations in order
        X_processed = DataLeakageHandler.drop_leakage_columns(X)
        X_processed = self.batch_engineer.transform(X_processed)
        X_processed = self.interaction_creator.transform(X_processed)
        X_processed = self.supplier_engineer.transform(X_processed)
        X_processed = self.timeseries_creator.transform(X_processed)
        X_processed = self.categorical_encoder.transform(X_processed)
        X_processed = self.anomaly_generator.transform(X_processed)

        X_processed = pd.DataFrame(
            self.imputer.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )

        X_processed = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )

        logger.info(f"[{datetime.now()}] Preprocessing complete.")
        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation"""
        return self.feature_names if self.feature_names is not None else []


if __name__ == "__main__":
    # Example usage
    logger.info("USG Preprocessing Pipeline initialized")
    logger.info("Use USGPreprocessingPipeline class for data transformation")
