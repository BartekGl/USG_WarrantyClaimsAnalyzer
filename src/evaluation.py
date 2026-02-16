"""
Model Evaluation and Validation Module
Comprehensive metrics, validation strategies, and robustness testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculation for binary classification
    Focuses on F1, PR-AUC, ROC-AUC, and business cost
    """

    def __init__(self, fn_cost: float = 1000, fp_cost: float = 50):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive set of metrics"""

        # Convert to numpy arrays to avoid index issues
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # Convert string labels to binary if needed
        if len(y_true_arr) > 0 and isinstance(y_true_arr[0], str):
            y_true_binary = np.array([1 if y == 'Yes' else 0 for y in y_true_arr])
        else:
            y_true_binary = y_true_arr

        if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
            y_pred_binary = np.array([1 if y == 'Yes' else 0 for y in y_pred_arr])
        else:
            y_pred_binary = y_pred_arr

        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }

        # Probability-based metrics
        if y_proba is not None:
            if len(y_proba.shape) > 1:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba

            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba_positive)
            metrics['pr_auc'] = average_precision_score(y_true_binary, y_proba_positive)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)

        # Business cost
        metrics['business_cost'] = (fn * self.fn_cost) + (fp * self.fp_cost)

        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info("=" * 60)
        logger.info(f"F1 Score:           {metrics['f1_score']:.4f}")
        logger.info(f"Precision:          {metrics['precision']:.4f}")
        logger.info(f"Recall:             {metrics['recall']:.4f}")
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")

        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            logger.info(f"PR-AUC:             {metrics['pr_auc']:.4f}")

        logger.info("-" * 60)
        logger.info(f"True Positives:     {metrics['true_positives']}")
        logger.info(f"False Positives:    {metrics['false_positives']}")
        logger.info(f"True Negatives:     {metrics['true_negatives']}")
        logger.info(f"False Negatives:    {metrics['false_negatives']}")
        logger.info("-" * 60)
        logger.info(f"Business Cost:      ${metrics['business_cost']:,.2f}")
        logger.info("=" * 60)


class CrossValidator:
    """
    Stratified K-Fold cross-validation with comprehensive metrics
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.cv_results = None

    def validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        logger.info(f"[{datetime.now()}] Starting {self.n_splits}-fold cross-validation...")

        # Encode target
        y_encoded = (y == 'Yes').astype(int) if isinstance(y.iloc[0], str) else y

        # Define scoring metrics
        scoring = {
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc'
        }

        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # Cross-validation
        cv_results = cross_validate(
            model, X, y_encoded,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        # Aggregate results
        results = {
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
        }

        self.cv_results = results

        logger.info(f"[{datetime.now()}] Cross-validation complete")
        logger.info(f"F1: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
        logger.info(f"ROC-AUC: {results['roc_auc_mean']:.4f} (+/- {results['roc_auc_std']:.4f})")

        return results


class ThresholdAnalyzer:
    """
    Analyzes precision-recall trade-offs at different thresholds
    Helps find optimal operating point
    """

    def __init__(self):
        self.thresholds = None
        self.precisions = None
        self.recalls = None
        self.f1_scores = None
        self.optimal_threshold = None

    def analyze(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Analyze thresholds and find optimal point"""

        # Convert to numpy arrays to avoid index issues
        y_true_arr = np.asarray(y_true)

        # Convert string labels to binary if needed
        if len(y_true_arr) > 0 and isinstance(y_true_arr[0], str):
            y_true_binary = np.array([1 if y == 'Yes' else 0 for y in y_true_arr])
        else:
            y_true_binary = y_true_arr

        if len(y_proba.shape) > 1:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba

        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true_binary, y_proba_positive)

        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        self.optimal_threshold = thresholds[optimal_idx]

        self.thresholds = thresholds
        self.precisions = precisions[:-1]
        self.recalls = recalls[:-1]
        self.f1_scores = f1_scores[:-1]

        results = {
            'optimal_threshold': float(self.optimal_threshold),
            'optimal_f1': float(f1_scores[optimal_idx]),
            'optimal_precision': float(precisions[optimal_idx]),
            'optimal_recall': float(recalls[optimal_idx])
        }

        logger.info(f"Optimal threshold: {results['optimal_threshold']:.4f}")
        logger.info(f"Optimal F1: {results['optimal_f1']:.4f}")

        # Plot if save path provided
        if save_path:
            self.plot_threshold_analysis(save_path)

        return results

    def plot_threshold_analysis(self, save_path: str):
        """Plot precision-recall-threshold curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Precision-Recall curve
        ax1.plot(self.recalls, self.precisions, linewidth=2)
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curve', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Metrics vs Threshold
        ax2.plot(self.thresholds, self.precisions, label='Precision', linewidth=2)
        ax2.plot(self.thresholds, self.recalls, label='Recall', linewidth=2)
        ax2.plot(self.thresholds, self.f1_scores, label='F1 Score', linewidth=2)
        ax2.axvline(self.optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Metrics vs Threshold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Threshold analysis plot saved to {save_path}")


class StressTestor:
    """
    Robustness testing with noise injection and feature permutation
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def noise_injection_test(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        noise_levels: List[float] = [0.05, 0.1, 0.15, 0.2]
    ) -> Dict[str, List[float]]:
        """Test model robustness to noise"""
        logger.info("Running noise injection stress test...")

        results = {
            'noise_level': [],
            'f1_score': [],
            'accuracy': []
        }

        # Encode target
        y_encoded = (y == 'Yes').astype(int) if isinstance(y.iloc[0], str) else y

        for noise_level in noise_levels:
            # Add Gaussian noise
            X_noisy = X.copy()
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X_noisy + noise

            # Predict
            y_pred = model.predict(X_noisy)
            y_pred_arr = np.asarray(y_pred)
            if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
                y_pred = np.array([1 if y == 'Yes' else 0 for y in y_pred_arr])
            else:
                y_pred = y_pred_arr

            # Calculate metrics
            f1 = f1_score(y_encoded, y_pred, zero_division=0)
            acc = accuracy_score(y_encoded, y_pred)

            results['noise_level'].append(noise_level)
            results['f1_score'].append(f1)
            results['accuracy'].append(acc)

            logger.info(f"Noise level {noise_level:.2f}: F1={f1:.4f}, Accuracy={acc:.4f}")

        return results

    def feature_permutation_test(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 10
    ) -> pd.DataFrame:
        """Test feature importance via permutation"""
        logger.info("Running feature permutation test...")

        # Encode target
        y_encoded = (y == 'Yes').astype(int) if isinstance(y.iloc[0], str) else y

        # Baseline performance
        y_pred = model.predict(X)
        y_pred_arr = np.asarray(y_pred)
        if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
            y_pred = np.array([1 if y == 'Yes' else 0 for y in y_pred_arr])
        else:
            y_pred = y_pred_arr
        baseline_f1 = f1_score(y_encoded, y_pred, zero_division=0)

        importance_scores = []

        for feature in X.columns:
            feature_scores = []

            for _ in range(n_iterations):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)

                # Predict
                y_pred = model.predict(X_permuted)
                y_pred_arr = np.asarray(y_pred)
                if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
                    y_pred = np.array([1 if y == 'Yes' else 0 for y in y_pred_arr])
                else:
                    y_pred = y_pred_arr

                # Calculate drop in performance
                permuted_f1 = f1_score(y_encoded, y_pred, zero_division=0)
                importance = baseline_f1 - permuted_f1
                feature_scores.append(importance)

            importance_scores.append({
                'feature': feature,
                'importance_mean': np.mean(feature_scores),
                'importance_std': np.std(feature_scores)
            })

        results = pd.DataFrame(importance_scores).sort_values('importance_mean', ascending=False)
        logger.info(f"Feature permutation test complete. Top feature: {results.iloc[0]['feature']}")

        return results


class FairnessChecker:
    """
    Checks model performance consistency across different subgroups
    """

    def __init__(self):
        self.fairness_results = {}

    def check_regional_fairness(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        region_column: str = 'Region'
    ) -> Dict[str, Dict[str, float]]:
        """Check performance consistency across regions"""
        if region_column not in X.columns:
            logger.warning(f"Column {region_column} not found. Skipping fairness check.")
            return {}

        logger.info("Checking regional fairness...")

        # Encode target
        y_encoded = (y == 'Yes').astype(int) if isinstance(y.iloc[0], str) else y

        regions = X[region_column].unique()
        results = {}

        for region in regions:
            mask = X[region_column] == region
            X_region = X[mask]
            y_region = y_encoded[mask]

            if len(y_region) < 10:  # Skip small groups
                continue

            # Predict
            y_pred = model.predict(X_region)
            y_pred_arr = np.asarray(y_pred)
            if len(y_pred_arr) > 0 and isinstance(y_pred_arr[0], str):
                y_pred = np.array([1 if y == 'Yes' else 0 for y in y_pred_arr])
            else:
                y_pred = y_pred_arr

            # Calculate metrics
            results[region] = {
                'f1_score': f1_score(y_region, y_pred, zero_division=0),
                'precision': precision_score(y_region, y_pred, zero_division=0),
                'recall': recall_score(y_region, y_pred, zero_division=0),
                'sample_size': int(len(y_region))
            }

            logger.info(f"{region}: F1={results[region]['f1_score']:.4f}, n={results[region]['sample_size']}")

        self.fairness_results = results
        return results


class ModelEvaluator:
    """
    Comprehensive model evaluation orchestrator
    Combines all evaluation components
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.metrics_calculator = MetricsCalculator()
        self.cross_validator = CrossValidator(seed=seed)
        self.threshold_analyzer = ThresholdAnalyzer()
        self.stress_testor = StressTestor(seed=seed)
        self.fairness_checker = FairnessChecker()

        self.evaluation_results = {}

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        run_stress_tests: bool = True,
        run_fairness_check: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        logger.info(f"[{datetime.now()}] Starting comprehensive model evaluation...")

        # 1. Basic metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
        self.metrics_calculator.print_metrics(metrics)
        self.evaluation_results['test_metrics'] = metrics

        # 2. Cross-validation (if training data provided)
        if X_train is not None and y_train is not None:
            try:
                logger.info("Running cross-validation...")
                cv_results = self.cross_validator.validate(model, X_train, y_train)
                self.evaluation_results['cv_results'] = cv_results
                logger.info("✓ Cross-validation complete")
            except ValueError as e:
                if "needs to have more than 1 class" in str(e):
                    logger.warning("⚠ Skipping cross-validation: Some folds have only one class due to data imbalance")
                    logger.warning("  Consider using the full dataset for cross-validation or adjusting fold count")
                    self.evaluation_results['cv_results'] = {
                        'status': 'skipped',
                        'reason': 'insufficient_class_distribution',
                        'message': 'Some CV folds had only one class - data too imbalanced for CV on this subset'
                    }
                else:
                    # Re-raise if it's a different ValueError
                    raise
            except Exception as e:
                logger.warning(f"⚠ Cross-validation failed: {e}")
                self.evaluation_results['cv_results'] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # 3. Threshold analysis
        threshold_results = self.threshold_analyzer.analyze(y_test, y_proba)
        self.evaluation_results['threshold_analysis'] = threshold_results

        # 4. Stress tests
        if run_stress_tests and X_test is not None:
            noise_results = self.stress_testor.noise_injection_test(model, X_test, y_test)
            self.evaluation_results['noise_injection'] = noise_results

        # 5. Fairness check
        if run_fairness_check and X_test is not None:
            fairness_results = self.fairness_checker.check_regional_fairness(
                model, X_test, y_test
            )
            self.evaluation_results['fairness'] = fairness_results

        logger.info(f"[{datetime.now()}] Evaluation complete")
        return self.evaluation_results

    def save_results(self, filepath: str):
        """Save evaluation results to JSON"""
        logger.info(f"Saving evaluation results to {filepath}")

        # Convert numpy types to native Python types
        results_serializable = self._convert_to_serializable(self.evaluation_results)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


if __name__ == "__main__":
    logger.info("Model Evaluation Module initialized")
    logger.info("Use ModelEvaluator class for comprehensive evaluation")
