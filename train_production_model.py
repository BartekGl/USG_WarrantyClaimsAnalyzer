"""
PHASE 1 (continued): Train Production Model
Trains a Balanced Random Forest on the 10,000 synthetic records.

Key characteristics:
- Model learns Cables-X as toxic supplier (~22% failure)
- Model learns Golden Zone (3.2-3.8s) as optimal
- Model learns WireTech as best cable supplier (~5% failure)

The trained model is saved for production inference.
CRITICAL: The real 2310 units are NEVER seen during training.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import BalancedRandomForestClassifier
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
SYNTHETIC_DATA_PATH = Path(__file__).parent / 'data' / 'synthetic' / 'synthetic_training_data.csv'
MODEL_OUTPUT_PATH = Path(__file__).parent / 'models'
RANDOM_SEED = 42

# Columns to exclude (leakage prevention)
LEAKAGE_COLUMNS = [
    'Warranty_Claim',  # Target
    'Claim_Type',      # Leakage
    'Repair_Cost_USD', # Leakage
    'Device_UUID',     # ID
    'Serial_Number',   # ID
    'Customer_Region', # Post-production
    'Installation_Date'  # Post-production
]


def load_synthetic_data(data_path: Path) -> pd.DataFrame:
    """Load synthetic training data."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found at {data_path}. "
            f"Run generate_training_data.py first."
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} synthetic training records")
    print(f"  Failure rate: {(df['Warranty_Claim'] == 'Yes').mean()*100:.1f}%")

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for training.
    Removes leakage columns and encodes categoricals.
    """
    # Identify columns to drop
    cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]

    # Create feature matrix
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df['Warranty_Claim']

    print(f"\nFeature preparation:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Dropped columns: {cols_to_drop}")

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Numerical features: {len(numerical_cols)}")

    # Encode categorical columns
    label_encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Encode target
    y_encoded = (y == 'Yes').astype(int)

    return X_encoded, y_encoded, label_encoders, X.columns.tolist()


def train_model(X: pd.DataFrame, y: pd.Series) -> BalancedRandomForestClassifier:
    """
    Train Balanced Random Forest model.

    Uses BalancedRandomForestClassifier to handle class imbalance.
    """
    print("\n" + "=" * 60)
    print("TRAINING BALANCED RANDOM FOREST")
    print("=" * 60)

    model = BalancedRandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        sampling_strategy='auto',
        replacement=True,
        n_jobs=-1,
        verbose=0
    )

    print(f"Model parameters:")
    print(f"  n_estimators: 200")
    print(f"  max_depth: 15")
    print(f"  sampling_strategy: auto (balanced)")

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Train final model
    print("\nTraining final model on full dataset...")
    model.fit(X, y)

    # Feature importances
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model


def validate_model_signals(model, X: pd.DataFrame, label_encoders: dict, feature_names: list):
    """
    Validate that the model has learned the key signals:
    1. Cables-X as toxic supplier
    2. Golden Zone for soldering time
    """
    print("\n" + "=" * 60)
    print("MODEL SIGNAL VALIDATION")
    print("=" * 60)

    # Test Cables-X effect
    if 'Cable_Harness_Supplier' in feature_names:
        print("\n1. Cable Supplier Effect Test:")

        # Create test samples - one with Cables-X, one with WireTech
        test_row = X.iloc[0:1].copy()

        # Find encoded values
        cable_le = label_encoders.get('Cable_Harness_Supplier')
        if cable_le:
            cables_x_encoded = None
            wiretech_encoded = None

            for cls in cable_le.classes_:
                if 'Cables-X' in cls:
                    cables_x_encoded = cable_le.transform([cls])[0]
                if 'WireTech' in cls:
                    wiretech_encoded = cable_le.transform([cls])[0]

            if cables_x_encoded is not None and wiretech_encoded is not None:
                # Test with Cables-X
                test_cables_x = test_row.copy()
                test_cables_x['Cable_Harness_Supplier'] = cables_x_encoded
                prob_cables_x = model.predict_proba(test_cables_x)[0, 1]

                # Test with WireTech
                test_wiretech = test_row.copy()
                test_wiretech['Cable_Harness_Supplier'] = wiretech_encoded
                prob_wiretech = model.predict_proba(test_wiretech)[0, 1]

                print(f"   Cables-X probability: {prob_cables_x*100:.1f}%")
                print(f"   WireTech probability: {prob_wiretech*100:.1f}%")
                print(f"   Difference: {(prob_cables_x - prob_wiretech)*100:.1f} percentage points")

                if prob_cables_x > prob_wiretech:
                    print("   SIGNAL LEARNED: Cables-X identified as higher risk")
                else:
                    print("   WARNING: Cables-X signal may not be strong enough")

    # Test Golden Zone effect
    if 'Soldering_Time_s' in feature_names:
        print("\n2. Soldering Time Effect Test:")

        test_row = X.iloc[0:1].copy()

        # Test at 3.5s (Golden Zone)
        test_golden = test_row.copy()
        test_golden['Soldering_Time_s'] = 3.5
        prob_golden = model.predict_proba(test_golden)[0, 1]

        # Test at 2.5s (outside Golden Zone)
        test_low = test_row.copy()
        test_low['Soldering_Time_s'] = 2.5
        prob_low = model.predict_proba(test_low)[0, 1]

        # Test at 4.5s (outside Golden Zone)
        test_high = test_row.copy()
        test_high['Soldering_Time_s'] = 4.5
        prob_high = model.predict_proba(test_high)[0, 1]

        print(f"   At 3.5s (Golden Zone): {prob_golden*100:.1f}%")
        print(f"   At 2.5s (below zone): {prob_low*100:.1f}%")
        print(f"   At 4.5s (above zone): {prob_high*100:.1f}%")

        if prob_golden < prob_low and prob_golden < prob_high:
            print("   SIGNAL LEARNED: Golden Zone identified as lower risk")
        else:
            print("   WARNING: Golden Zone signal may not be strong enough")


def save_model(
    model: BalancedRandomForestClassifier,
    label_encoders: dict,
    feature_names: list,
    output_dir: Path
):
    """Save trained model and preprocessor."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'production_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessor (label encoders + metadata)
    preprocessor = {
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'categorical_cols': [col for col, le in label_encoders.items()],
        'model_type': 'BalancedRandomForestClassifier',
        'training_samples': 10000,
        'signals': ['Cables-X toxic supplier', 'Golden Zone 3.2-3.8s']
    }

    preprocessor_path = output_dir / 'production_preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")

    # Save feature names as JSON for reference
    feature_path = output_dir / 'feature_names.json'
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {feature_path}")

    return model_path, preprocessor_path


def main():
    """Main entry point for model training."""
    print("\n" + "=" * 80)
    print("PHASE 1: TRAIN PRODUCTION MODEL")
    print("=" * 80)
    print("Training on 10,000 synthetic records")
    print("Real 2310 units are NOT included in training")

    # Step 1: Load synthetic data
    df = load_synthetic_data(SYNTHETIC_DATA_PATH)

    # Step 2: Prepare features
    X, y, label_encoders, feature_names = prepare_features(df)

    # Step 3: Train model
    model = train_model(X, y)

    # Step 4: Validate signals
    validate_model_signals(model, X, label_encoders, feature_names)

    # Step 5: Save model
    model_path, preprocessor_path = save_model(
        model, label_encoders, feature_names, MODEL_OUTPUT_PATH
    )

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")
    print("\nNext step: Run the dashboard with real production data")

    return model_path, preprocessor_path


if __name__ == "__main__":
    main()
