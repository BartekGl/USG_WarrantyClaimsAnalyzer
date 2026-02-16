"""
Standalone Training Script for USG Failure Prediction
Run this script directly to train the model without notebooks
"""

import sys
import os
from pathlib import Path

# Add src to path
# Go up from scripts/ to project root, then add src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from preprocessing import USGPreprocessingPipeline
from model import USGFailurePredictionModel, XGBoostOptimizer
from evaluation import ModelEvaluator

from sklearn.model_selection import train_test_split


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'data/processed',
        'models',
        'reports/visualizations',
        'reports/metrics'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified")


def load_and_preprocess_data(data_path: str, force_reprocess: bool = False):
    """Load and preprocess data"""
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)

    processed_X_path = 'data/processed/X_processed.csv'
    processed_y_path = 'data/processed/y_target.csv'
    preprocessor_path = 'models/preprocessor.pkl'

    # Try to load processed data if available
    if not force_reprocess and Path(processed_X_path).exists() and Path(processed_y_path).exists():
        print("\nLoading pre-processed data...")
        X = pd.read_csv(processed_X_path)
        y = pd.read_csv(processed_y_path).squeeze()
        preprocessor = joblib.load(preprocessor_path)
        print(f"✓ Loaded processed data: X={X.shape}, y={y.shape}")
    else:
        print("\nProcessing raw data...")

        # Check if data file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"\n❌ Data file not found at: {data_path}\n"
                f"Please place your USG_Data_cleared.csv file in the data/raw/ directory"
            )

        # Load raw data
        df = pd.read_csv(data_path)
        print(f"✓ Loaded raw data: {df.shape}")

        # Check for target column
        if 'Warranty_Claim' not in df.columns:
            raise ValueError("Dataset must contain 'Warranty_Claim' column")

        # Separate features and target
        X_raw = df.drop('Warranty_Claim', axis=1)
        y = df['Warranty_Claim']

        # Preprocess
        print("\nApplying feature engineering pipeline...")
        preprocessor = USGPreprocessingPipeline(seed=42)
        X = preprocessor.fit_transform(X_raw, y)

        # Save processed data
        print("\nSaving processed data...")
        X.to_csv(processed_X_path, index=False)
        y.to_csv(processed_y_path, index=False)
        joblib.dump(preprocessor, preprocessor_path)

        print(f"✓ Data processed and saved: X={X.shape}, y={y.shape}")

    # Display target distribution
    print("\nTarget Distribution:")
    print(y.value_counts())
    failure_rate = (y == 'Yes').mean() * 100
    print(f"Failure rate: {failure_rate:.2f}%")

    return X, y, preprocessor


def split_data(X, y):
    """Split data into train and test sets"""
    print("\n" + "="*60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nTrain target distribution:")
    print(y_train.value_counts())
    print(f"\nTest target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    print("\n" + "="*60)
    print("STEP 3: HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    print(f"\nRunning Optuna optimization with {n_trials} trials...")
    print("This may take 3-5 minutes...\n")

    optimizer = XGBoostOptimizer(
        n_trials=n_trials,
        cv_folds=5,
        seed=42,
        use_smote=True
    )

    best_params = optimizer.optimize(X_train, y_train)

    print("\n" + "="*60)
    print("OPTIMAL HYPERPARAMETERS")
    print("="*60)
    for param, value in best_params.items():
        print(f"{param:20s}: {value}")
    print("="*60)
    print(f"Best CV F1 Score: {optimizer.study.best_value:.4f}")
    print("="*60)

    # Save best parameters
    with open('reports/metrics/best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    return best_params


def train_model(X_train, y_train, best_params):
    """Train the full ensemble model"""
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)

    print("\nTraining ensemble model (XGBoost + Random Forest + LightGBM)...")

    model = USGFailurePredictionModel(
        optimize_hyperparams=False,
        use_ensemble=True,
        use_calibration=True,
        seed=42
    )

    model.best_params = best_params
    model.fit(X_train, y_train)

    print(f"\n✓ Model training complete in {model.training_time:.2f} seconds")

    # Save model
    model.save('models/model.pkl')
    print("✓ Model saved to models/model.pkl")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)

    evaluator = ModelEvaluator(seed=42)

    print("\nRunning comprehensive evaluation...")
    evaluation_results = evaluator.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        run_stress_tests=True,
        run_fairness_check=True
    )

    # Display results
    test_metrics = evaluation_results['test_metrics']

    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"F1 Score:      {test_metrics['f1_score']:.4f}")
    print(f"Precision:     {test_metrics['precision']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:        {test_metrics['pr_auc']:.4f}")
    print("="*60)

    print("\nConfusion Matrix:")
    print(f"  TP: {test_metrics['true_positives']:4d}  |  FP: {test_metrics['false_positives']:4d}")
    print(f"  FN: {test_metrics['false_negatives']:4d}  |  TN: {test_metrics['true_negatives']:4d}")
    print(f"\nBusiness Cost: ${test_metrics['business_cost']:,.2f}")

    # Cross-validation results
    if 'cv_results' in evaluation_results:
        cv = evaluation_results['cv_results']
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS (5-Fold)")
        print("="*60)

        # Check if CV was actually performed or skipped
        if 'status' in cv and cv['status'] == 'skipped':
            print("⚠ Cross-validation was skipped")
            print(f"Reason: {cv.get('reason', 'Unknown')}")
            if 'message' in cv:
                print(f"Message: {cv['message']}")
        elif 'f1_mean' in cv:
            # CV was successful, print results
            print(f"F1:        {cv['f1_mean']:.4f} (+/- {cv['f1_std']:.4f})")
            print(f"Precision: {cv['precision_mean']:.4f} (+/- {cv['precision_std']:.4f})")
            print(f"Recall:    {cv['recall_mean']:.4f} (+/- {cv['recall_std']:.4f})")
            print(f"ROC-AUC:   {cv['roc_auc_mean']:.4f} (+/- {cv['roc_auc_std']:.4f})")
        else:
            print("⚠ Cross-validation results unavailable")

        print("="*60)

    # Save evaluation results
    evaluator.save_results('reports/metrics/evaluation_results.json')
    print("\n✓ Evaluation results saved to reports/metrics/evaluation_results.json")

    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("✓ Feature names saved to models/feature_names.json")

    return evaluation_results


def main():
    """Main training pipeline"""
    start_time = datetime.now()

    print("="*60)
    print("USG FAILURE PREDICTION - MODEL TRAINING")
    print("="*60)
    print(f"Started: {start_time}")
    print("="*60)

    try:
        # Step 1: Create directories
        create_directories()

        # Step 2: Load and preprocess data
        data_path = 'data/raw/USG_Data_cleared.csv'
        X, y, preprocessor = load_and_preprocess_data(data_path, force_reprocess=False)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Step 4: Optimize hyperparameters
        best_params = optimize_hyperparameters(X_train, y_train, n_trials=50)

        # Step 5: Train model
        model = train_model(X_train, y_train, best_params)

        # Step 6: Evaluate model
        feature_names = X.columns.tolist()
        evaluation_results = evaluate_model(
            model, X_train, y_train, X_test, y_test, feature_names
        )

        # Final summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTotal Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nModel saved to: models/model.pkl")
        print(f"Preprocessor saved to: models/preprocessor.pkl")
        print(f"Results saved to: reports/metrics/")

        test_metrics = evaluation_results['test_metrics']
        print(f"\nFinal Performance:")
        print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  - PR-AUC: {test_metrics['pr_auc']:.4f}")

        print("\n" + "="*60)
        print("Next Steps:")
        print("  1. Review SHAP analysis: python scripts/run_shap_analysis.py")
        print("  2. Start API server: uvicorn src.api:app --reload")
        print("  3. View dashboard: cd frontend && npm run dev")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
