"""
Analytics Engine for USG Failure Prediction Dashboard
Handles SHAP interpretability, clustering, model comparison, and advanced visualizations.

IMPORTANT: This module includes a "sniper patch" to ensure compatibility between
SHAP's TreeExplainer and XGBoost 2.0+. The patch temporarily overrides the float()
built-in function within SHAP's execution context to handle bracketed base_score
values (e.g., "[8.3466375E-1]") that cause ValueError during parsing.

When SHAP's XGBTreeModelLoader calls float("[8.3466375E-1]"), our patched version
strips the brackets first, preventing the ValueError. The patch is applied only
during TreeExplainer initialization and is automatically removed to prevent side effects.
"""

import sys
from pathlib import Path

# Add project root and src to path for imports during unpickling
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))  # For ml_core.py in root
sys.path.insert(0, str(project_root / 'src'))  # For legacy src modules

import numpy as np
import pandas as pd
import joblib
import json
import builtins
from typing import Dict, Tuple, Optional, List
from contextlib import contextmanager

import shap
import shap.explainers._tree as shap_tree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix
)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# CSV Loader (delimiter-safe)
# ----------------------------------------------------------------------------
def read_usg_csv(path):
    """
    Read USG CSV with flexible delimiter handling.

    Supports both comma- and semicolon-delimited files.
    Raises a clear error if Warranty_Claim is still missing.
    """
    # Try auto-detect first (handles comma/semicolon in most cases)
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    # If pandas failed to split columns, try semicolon explicitly
    if 'Warranty_Claim' not in df.columns and len(df.columns) == 1:
        header = df.columns[0]
        if isinstance(header, str) and ';' in header:
            df = pd.read_csv(path, sep=';', encoding="utf-8-sig")

    if 'Warranty_Claim' not in df.columns:
        raise ValueError(
            "Required column 'Warranty_Claim' not found in dataset. "
            f"Columns present: {list(df.columns)[:10]}..."
        )

    return df

# Import custom classes needed for unpickling
try:
    from ml_core import LabelEncoderPipeline
except ImportError:
    # Fallback: create placeholder class
    class LabelEncoderPipeline:
        """Placeholder for unpickling preprocessor.pkl"""
        pass

# Pickle compatibility: Handle __main__.LabelEncoderPipeline redirects
# When ml_core.py is run as a script, pickle saves classes as __main__.ClassName
# We need to make them available for unpickling in different contexts
import pickle
import io

class CompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects __main__ references to proper modules"""
    def find_class(self, module, name):
        # Redirect __main__ classes to their proper modules
        if module == '__main__':
            if name == 'LabelEncoderPipeline':
                module = 'ml_core'
            elif name == 'USGFailurePredictionModel':
                module = 'src.model'
        return super().find_class(module, name)

def load_pickle_with_compat(filepath):
    """
    Load pickle file with __main__ compatibility.

    IMPORTANT: When ml_core.py is run as a script, pickle saves classes as
    __main__.ClassName. This function ensures LabelEncoderPipeline is available
    in __main__ namespace before loading, so joblib can resolve the class.
    """
    import __main__
    import sys

    # CRITICAL: Import ml_core FIRST to ensure LabelEncoderPipeline is available
    # This must happen before joblib.load() which uses standard pickle internally
    try:
        import ml_core
        __main__.LabelEncoderPipeline = ml_core.LabelEncoderPipeline
        sys.modules['__main__'].LabelEncoderPipeline = ml_core.LabelEncoderPipeline
    except ImportError:
        __main__.LabelEncoderPipeline = LabelEncoderPipeline

    # Use joblib.load which handles numpy arrays better than raw pickle
    return joblib.load(filepath)


# ============================================================================
# MONKEY PATCH FOR SHAP + XGBoost 2.0+ COMPATIBILITY
# ============================================================================

def _safe_float_cast(value):
    """
    Safe float conversion that handles XGBoost 2.0+ bracketed strings.

    XGBoost 2.0+ serializes base_score as "[8.3466375E-1]" which causes
    ValueError when SHAP's XGBTreeModelLoader tries: float("[8.3466375E-1]")

    This function strips brackets before conversion:
        "[8.3466375E-1]" -> "8.3466375E-1" -> 0.83466375

    Args:
        value: Value to convert to float (can be string, number, etc.)

    Returns:
        float: Converted value

    Raises:
        ValueError: If value cannot be converted to float
    """
    original_float = builtins.float

    if isinstance(value, str):
        # Strip brackets and whitespace from bracketed scientific notation
        cleaned = value.strip('[]').strip()
        return original_float(cleaned)
    else:
        # Pass through non-string values unchanged
        return original_float(value)


@contextmanager
def _patch_shap_float():
    """
    Context manager that patches SHAP's float() built-in reference.

    This is the "sniper patch" - the most surgical fix possible. It directly
    replaces the float() built-in within SHAP's namespace to handle bracketed
    strings from XGBoost 2.0+.

    Why this works when json.loads patching doesn't:
    - SHAP's XGBTreeModelLoader calls float() directly on parsed JSON values
    - By the time json.loads returns, the data is already parsed
    - We need to intercept the float() call itself, not the JSON parsing

    The patch:
    1. Imports shap.explainers._tree to access SHAP's namespace
    2. Saves the original float reference from SHAP's builtins
    3. Replaces it with _safe_float_cast that strips brackets
    4. Restores the original after SHAP initialization completes

    Target: shap.explainers._tree module's float() calls
    Duration: Only during TreeExplainer initialization

    Usage:
        with _patch_shap_float():
            explainer = shap.TreeExplainer(model, data)
    """
    import builtins

    try:
        # Import SHAP's internal tree module
        import shap.explainers._tree as shap_tree

        # Save SHAP's original float reference
        # SHAP's module uses builtins.float, so we need to patch at module level
        original_float = shap_tree.float if hasattr(shap_tree, 'float') else builtins.float

        # Create wrapper that uses the safe float cast
        def patched_float(value):
            """Patched float that handles bracketed strings."""
            return _safe_float_cast(value)

        # Apply the patch to SHAP's namespace
        # We patch the module's __builtins__ if accessible, or inject directly
        try:
            # Try to patch at the module level
            if hasattr(shap_tree, '__builtins__'):
                if isinstance(shap_tree.__builtins__, dict):
                    shap_tree.__builtins__['float'] = patched_float
                else:
                    # It's a module
                    shap_tree.__builtins__.float = patched_float

            # Also patch builtins globally for safety (SHAP will see this)
            builtins.float = patched_float

        except (AttributeError, TypeError):
            # If we can't patch __builtins__, just patch builtins globally
            builtins.float = patched_float

        try:
            yield
        finally:
            # Always restore original float
            try:
                if hasattr(shap_tree, '__builtins__'):
                    if isinstance(shap_tree.__builtins__, dict):
                        shap_tree.__builtins__['float'] = original_float
                    else:
                        shap_tree.__builtins__.float = original_float
            except (AttributeError, TypeError):
                pass

            # Restore global builtins
            builtins.float = original_float

    except (ImportError, AttributeError) as e:
        # If we can't import SHAP or access its structure, proceed without patching
        print(f"⚠ Warning: Could not patch SHAP's float(): {e}")
        yield


# ============================================================================
# END MONKEY PATCH
# ============================================================================


class AnalyticsEngine:
    def __init__(
        self,
        model_path='models/model.pkl',
        preprocessor_path='models/preprocessor.pkl',
        data_path='data/raw/USG_Data_cleared.csv'
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.data_path = Path(data_path)

        self.model = None
        self.preprocessor = None
        self.shap_explainer = None
        self.rf_model = None
        self.pca_model = None

        self.X = None
        self.y = None
        self.X_processed = None
        self.feature_names = None

        self._load_models()
        self._load_data()

        # Initialize SHAP explainer automatically
        print("🔄 Initializing SHAP explainer...")
        self.initialize_shap(max_samples=50)

    def _load_models(self):
        if self.model_path.exists():
            # Use compatibility unpickler to handle __main__ module redirects
            try:
                self.model = load_pickle_with_compat(self.model_path)
                print(f"✓ Model loaded from {self.model_path} (with compatibility)")
            except Exception as e:
                # Fallback to joblib.load if custom unpickler fails
                print(f"⚠ Compatibility unpickler failed: {e}, trying joblib.load...")
                self.model = joblib.load(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if self.preprocessor_path.exists():
            # Use compatibility unpickler to handle __main__ module redirects
            try:
                self.preprocessor = load_pickle_with_compat(self.preprocessor_path)
                print(f"✓ Preprocessor loaded from {self.preprocessor_path} (with compatibility)")
            except Exception as e:
                # Fallback to joblib.load if custom unpickler fails
                print(f"⚠ Compatibility unpickler failed: {e}, trying joblib.load...")
                self.preprocessor = joblib.load(self.preprocessor_path)
                print(f"✓ Preprocessor loaded from {self.preprocessor_path}")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

    def _extract_feature_names_from_preprocessor(self, preprocessor, input_features):
        """
        Extract real feature names from preprocessor, handling ColumnTransformer.

        Args:
            preprocessor: The fitted preprocessor (Pipeline, ColumnTransformer, etc.)
            input_features: Original input feature names

        Returns:
            List of feature names after transformation
        """
        try:
            # Try the standard sklearn 1.0+ method first
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = list(preprocessor.get_feature_names_out())
                print(f"  ✓ Extracted {len(feature_names)} feature names via get_feature_names_out()")
                return feature_names
        except Exception as e:
            print(f"  ⚠ get_feature_names_out() failed: {e}")

        # Handle ColumnTransformer manually
        from sklearn.compose import ColumnTransformer

        if isinstance(preprocessor, ColumnTransformer):
            print(f"  → Detected ColumnTransformer, extracting from internal transformers...")
            feature_names = []

            for name, transformer, columns in preprocessor.transformers_:
                if name == 'remainder' or transformer == 'drop':
                    continue

                # Get column names for this transformer
                if isinstance(columns, slice):
                    cols = input_features[columns]
                elif isinstance(columns, (list, np.ndarray)):
                    cols = [input_features[i] if isinstance(i, int) else i for i in columns]
                else:
                    cols = columns

                # Handle different transformer types
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        trans_features = transformer.get_feature_names_out(cols)
                        feature_names.extend(trans_features)
                        print(f"    → {name}: {len(trans_features)} features")
                    except:
                        # Fallback to original column names
                        if isinstance(cols, (list, np.ndarray)):
                            feature_names.extend(cols)
                        print(f"    → {name}: using original columns")
                else:
                    # No feature name extraction, use original columns
                    if isinstance(cols, (list, np.ndarray)):
                        feature_names.extend(cols)
                    print(f"    → {name}: using original columns")

            print(f"  ✓ Extracted {len(feature_names)} total features from ColumnTransformer")
            return feature_names

        # Fallback: return original input feature names
        print(f"  ⚠ Could not extract feature names, using original input features")
        return list(input_features)

    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        df = read_usg_csv(self.data_path)
        self.y = df['Warranty_Claim']
        self.X = df.drop('Warranty_Claim', axis=1)

        print(f"✓ Raw data loaded: {self.X.shape[0]:,} samples, {self.X.shape[1]} original features")
        print(f"  Original columns: {self.X.columns.tolist()[:5]}..." if len(self.X.columns) > 5 else f"  Original columns: {self.X.columns.tolist()}")

        # Transform and convert to numpy array for consistent indexing
        X_transformed = self.preprocessor.transform(self.X)
        if hasattr(X_transformed, 'toarray'):
            # Handle sparse matrices
            self.X_processed = X_transformed.toarray()
        elif isinstance(X_transformed, pd.DataFrame):
            # Convert DataFrame to numpy array to avoid indexing issues
            self.X_processed = X_transformed.values
        else:
            # Already numpy array
            self.X_processed = X_transformed

        print(f"✓ Data preprocessed: shape {self.X_processed.shape}")

        # Extract real feature names from preprocessor
        print(f"→ Extracting feature names from preprocessor...")
        self.feature_names = self._extract_feature_names_from_preprocessor(
            self.preprocessor,
            self.X.columns.tolist()
        )

        # CRITICAL SAFETY CHECK: Ensure feature_names matches X_processed shape
        if len(self.feature_names) != self.X_processed.shape[1]:
            print(f"⚠️ WARNING: Feature name count mismatch!")
            print(f"  → feature_names extracted: {len(self.feature_names)}")
            print(f"  → X_processed columns: {self.X_processed.shape[1]}")
            print(f"  → Padding/truncating to match X_processed shape")

            if len(self.feature_names) < self.X_processed.shape[1]:
                # Need more names - pad with generic names
                num_missing = self.X_processed.shape[1] - len(self.feature_names)
                padding = [f"Feature_{i}" for i in range(len(self.feature_names), self.X_processed.shape[1])]
                self.feature_names.extend(padding)
                print(f"  ✓ Padded with {num_missing} generic names")
            else:
                # Too many names - truncate
                self.feature_names = self.feature_names[:self.X_processed.shape[1]]
                print(f"  ✓ Truncated to {self.X_processed.shape[1]} features")

        print(f"✓ Final feature names: {len(self.feature_names)} features")
        print(f"  Sample names: {self.feature_names[:5]}..." if len(self.feature_names) > 5 else f"  Feature names: {self.feature_names}")
        print(f"  → self.y shape: {len(self.y)}")
        print(f"  → self.X_processed shape: {self.X_processed.shape}")
        print(f"  → feature_names verified: {len(self.feature_names)} matches columns")

    def _is_tree_based_model(self, model) -> bool:
        """
        Check if a model is a known tree-based estimator compatible with SHAP TreeExplainer.

        Args:
            model: Model object to check

        Returns:
            True if the model is a recognized tree-based estimator
        """
        model_class_name = type(model).__name__
        model_module = type(model).__module__

        # Known tree-based models
        tree_indicators = [
            'XGBClassifier', 'XGBRegressor', 'XGBModel', 'XGBRFClassifier', 'XGBRFRegressor',
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'LGBMClassifier', 'LGBMRegressor', 'LGBMModel',
            'CatBoostClassifier', 'CatBoostRegressor'
        ]

        # Check class name
        if any(indicator in model_class_name for indicator in tree_indicators):
            return True

        # Check if from xgboost, lightgbm, or catboost packages
        if any(pkg in model_module for pkg in ['xgboost', 'lightgbm', 'catboost']):
            return True

        # Check for key XGBoost method
        if hasattr(model, 'get_booster') and callable(getattr(model, 'get_booster')):
            return True

        return False

    def initialize_shap(self, max_samples=50):
        """
        Initialize SHAP TreeExplainer with deep recursive model unwrapping.

        This method aggressively unwraps nested model wrappers to find the raw
        tree-based estimator (XGBClassifier, RandomForestClassifier, etc.) that
        TreeExplainer requires. It handles multiple wrapper patterns including:
        - Custom wrappers (USGFailurePredictionModel via .model)
        - CalibratedClassifierCV (via .calibrated_classifiers_)
        - CalibratedModel (via .base_model) - CRITICAL for this project
        - VotingClassifier/Ensembles (via .estimators_[0]) - extracts first estimator
        - sklearn wrapper patterns (.estimator, .base_estimator, .base_estimator_)

        Unwrapping chain example:
        USGFailurePredictionModel -> .model -> CalibratedModel -> .base_model
        -> VotingClassifier -> .estimators_[0] -> XGBClassifier

        XGBoost 2.0+ Compatibility - The "Sniper Patch":
        Applies a multi-layer fix for XGBoost 2.0+ compatibility:
        1. Cleans and reloads booster config JSON (removes brackets from base_score)
        2. Sets base_score parameter directly on the booster as backup
        3. Applies "sniper patch" to shap_tree.float during TreeExplainer initialization
           - Temporarily overrides shap.explainers._tree.float with safe_float
           - Intercepts float() calls to strip brackets: "[8.3466375E-1]" -> 0.83466375
           - Targets the exact point of failure in XGBTreeModelLoader

        The sniper patch works where json.loads patching fails because:
        - SHAP calls float() directly on already-parsed values
        - Patching shap_tree.float intercepts at the conversion point, not parsing
        - Most surgical fix possible - patches only SHAP's float reference

        The patch is safely removed via try...finally immediately after initialization.

        Args:
            max_samples: Number of background samples for SHAP (default: 50)

        Returns:
            Initialized SHAP TreeExplainer

        Raises:
            RuntimeError: If unwrapping fails to find a compatible tree-based model
        """
        if self.shap_explainer is None:
            print("=" * 70)
            print("Initializing SHAP with deep recursive model unwrapping...")
            print("=" * 70)
            print(f"Initial model type: {type(self.model).__name__}")
            print(f"Initial model module: {type(self.model).__module__}")

            # 1. Deep recursive unwrapping to find the raw tree-based estimator
            shap_model = self.model
            unwrap_path = [type(self.model).__name__]
            max_unwrap_iterations = 15  # Increased safety limit
            iteration = 0

            while iteration < max_unwrap_iterations:
                # Check if we've reached a tree-based model
                if self._is_tree_based_model(shap_model):
                    print(f"✓ Found tree-based model: {type(shap_model).__name__}")
                    break

                unwrapped = False
                current_type = type(shap_model).__name__

                # Priority 1: CalibratedClassifierCV - most common wrapper
                if hasattr(shap_model, 'calibrated_classifiers_') and len(shap_model.calibrated_classifiers_) > 0:
                    print(f"  [{iteration}] Unwrapping CalibratedClassifierCV...")
                    calibrated_clf = shap_model.calibrated_classifiers_[0]

                    # Try to get base_estimator first (deeper), then estimator
                    if hasattr(calibrated_clf, 'base_estimator'):
                        shap_model = calibrated_clf.base_estimator
                        print(f"      → Found via calibrated_classifiers_[0].base_estimator")
                    elif hasattr(calibrated_clf, 'estimator'):
                        shap_model = calibrated_clf.estimator
                        print(f"      → Found via calibrated_classifiers_[0].estimator")
                    else:
                        # Fallback: use the calibrated classifier itself
                        shap_model = calibrated_clf
                        print(f"      → Using calibrated_classifiers_[0] directly")

                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 2: base_model (CalibratedModel pattern - CRITICAL)
                elif hasattr(shap_model, 'base_model') and shap_model.base_model is not None:
                    print(f"  [{iteration}] Unwrapping via .base_model (CalibratedModel)...")
                    shap_model = shap_model.base_model
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 3: estimators_ (VotingClassifier/ensemble pattern)
                elif hasattr(shap_model, 'estimators_') and len(shap_model.estimators_) > 0:
                    print(f"  [{iteration}] Unwrapping VotingClassifier/Ensemble via .estimators_[0]...")
                    shap_model = shap_model.estimators_[0]
                    print(f"      → Extracted first estimator from ensemble: {type(shap_model).__name__}")
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 4: base_estimator (sklearn meta-estimator pattern)
                elif hasattr(shap_model, 'base_estimator') and shap_model.base_estimator is not None:
                    print(f"  [{iteration}] Unwrapping via .base_estimator...")
                    shap_model = shap_model.base_estimator
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 5: base_estimator_ (fitted meta-estimators)
                elif hasattr(shap_model, 'base_estimator_') and shap_model.base_estimator_ is not None:
                    print(f"  [{iteration}] Unwrapping via .base_estimator_...")
                    shap_model = shap_model.base_estimator_
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 6: estimator (standard sklearn wrapper)
                elif hasattr(shap_model, 'estimator') and hasattr(shap_model.estimator, '__class__'):
                    print(f"  [{iteration}] Unwrapping via .estimator...")
                    shap_model = shap_model.estimator
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 7: .model (custom wrapper pattern)
                elif hasattr(shap_model, 'model') and not isinstance(shap_model.model, (pd.DataFrame, np.ndarray, type(None))):
                    print(f"  [{iteration}] Unwrapping via .model...")
                    shap_model = shap_model.model
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 8: _model (private attribute pattern)
                elif hasattr(shap_model, '_model') and not isinstance(shap_model._model, (pd.DataFrame, np.ndarray, type(None))):
                    print(f"  [{iteration}] Unwrapping via ._model...")
                    shap_model = shap_model._model
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                # Priority 9: _estimator (alternative private attribute)
                elif hasattr(shap_model, '_estimator') and shap_model._estimator is not None:
                    print(f"  [{iteration}] Unwrapping via ._estimator...")
                    shap_model = shap_model._estimator
                    unwrap_path.append(type(shap_model).__name__)
                    unwrapped = True

                if not unwrapped:
                    print(f"  [{iteration}] No more unwrapping possible for {current_type}")
                    break

                iteration += 1

            print("=" * 70)
            print(f"✓ Unwrapping complete after {iteration} iterations")
            print(f"✓ Unwrapping path: {' -> '.join(unwrap_path)}")
            print(f"✓ Final model type: {type(shap_model).__name__}")
            print(f"✓ Final model module: {type(shap_model).__module__}")
            print("=" * 70)

            # Validate that we found a tree-based model
            if not self._is_tree_based_model(shap_model):
                print(f"\n❌ VALIDATION FAILED: Final model is not a recognized tree-based estimator")
                print(f"   Model type: {type(shap_model).__name__}")
                print(f"   Model module: {type(shap_model).__module__}")
                print(f"   Available attributes: {[attr for attr in dir(shap_model) if not attr.startswith('_')][:15]}")

                raise RuntimeError(
                    f"Model unwrapping failed to find a tree-based estimator. "
                    f"Final type: {type(shap_model).__name__}. "
                    f"SHAP TreeExplainer requires XGBoost, RandomForest, or similar models."
                )

            print("✓ Validation passed: Model is tree-based and compatible with SHAP")

            # 2. Fix for XGBoost 2.0+ JSON formatting bug (only for XGBoost models)
            if hasattr(shap_model, 'get_booster'):
                try:
                    print("\n" + "-" * 70)
                    print("Applying XGBoost 2.0+ multi-layer fix...")
                    print("-" * 70)
                    booster = shap_model.get_booster()
                    config = json.loads(booster.save_config())

                    # Track extracted base_score for direct parameter setting
                    extracted_base_score = None

                    def clean_config_recursive(obj, path=""):
                        """Recursively clean XGBoost config, fixing base_score formatting."""
                        nonlocal extracted_base_score
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                current_path = f"{path}.{k}" if path else k
                                if k == 'base_score' and isinstance(v, str):
                                    # Remove brackets and whitespace from "[5E-1]" -> "5E-1"
                                    if '[' in v or ']' in v:
                                        cleaned = v.strip('[]').strip()
                                        print(f"  Config fix: {current_path}: '{v}' -> '{cleaned}'")
                                        obj[k] = cleaned
                                        # Extract numeric value for direct parameter setting
                                        try:
                                            extracted_base_score = float(cleaned)
                                        except ValueError:
                                            pass
                                else:
                                    clean_config_recursive(v, current_path)
                        elif isinstance(obj, list):
                            for idx, item in enumerate(obj):
                                clean_config_recursive(item, f"{path}[{idx}]")

                    clean_config_recursive(config)
                    booster.load_config(json.dumps(config))
                    print("  ✓ Step 1: Booster config JSON cleaned and reloaded")

                    # Additional backup: Set base_score parameter directly on booster
                    if extracted_base_score is not None:
                        try:
                            booster.set_param('base_score', str(extracted_base_score))
                            print(f"  ✓ Step 2: Set base_score parameter directly: {extracted_base_score}")
                        except Exception as param_error:
                            print(f"  ⚠ Step 2 skipped: Could not set base_score param: {param_error}")

                    print("✓ XGBoost multi-layer fix complete")
                    print("-" * 70)
                except Exception as e:
                    print(f"⚠ XGBoost fix skipped (non-critical): {e}")
                    print("-" * 70)

            # 3. Prepare background data for SHAP
            try:
                sample_size = min(max_samples, len(self.X_processed))
                sample_idx = np.random.choice(
                    len(self.X_processed),
                    size=sample_size,
                    replace=False
                )
                background = self.X_processed[sample_idx]
                print(f"\n✓ Background data prepared: {sample_size} samples")
            except Exception as e:
                print(f"❌ Failed to prepare background data: {e}")
                raise

            # 4. Initialize SHAP TreeExplainer with safe_float patch
            try:
                print(f"\n" + "=" * 70)
                print(f"Creating SHAP TreeExplainer...")
                print(f"  Model type: {type(shap_model)}")
                print(f"  Background shape: {background.shape}")
                print("-" * 70)
                print("⚠ APPLYING SNIPER PATCH TO SHAP NAMESPACE")
                print("  Target: shap.explainers._tree.float")
                print("  Purpose: Intercept float() calls to strip brackets from strings")
                print("  Method: Override shap_tree.float with safe_float")
                print("  This is the most surgical fix - patches at the point of failure")
                print("  Patch will be automatically removed after initialization")
                print("=" * 70)

                # Define safe_float converter
                def safe_float(x):
                    """Safe float converter that handles bracketed strings."""
                    if isinstance(x, str) and '[' in x:
                        return float(x.strip('[]'))
                    return float(x)

                # Save original shap_tree.float reference
                original_shap_float = getattr(shap_tree, 'float', float)

                try:
                    # Temporarily override shap_tree.float with safe_float
                    shap_tree.float = safe_float

                    # Initialize SHAP TreeExplainer
                    self.shap_explainer = shap.TreeExplainer(shap_model, background)

                finally:
                    # Always restore original float reference
                    shap_tree.float = original_shap_float

                print("✓ SHAP TreeExplainer successfully initialized!")
                print("✓ Sniper patch removed - shap_tree.float restored to original")
                print("=" * 70)
            except Exception as e:
                print(f"\n{'=' * 70}")
                print("❌ CRITICAL: SHAP TreeExplainer initialization failed")
                print("=" * 70)
                print(f"Error: {e}")
                print(f"Model type: {type(shap_model)}")
                print(f"Model class: {type(shap_model).__name__}")
                print(f"Model module: {type(shap_model).__module__}")

                # Enhanced debugging
                if hasattr(shap_model, 'get_booster'):
                    print("✓ Model has get_booster() method (XGBoost detected)")
                if hasattr(shap_model, 'estimators_'):
                    print("✓ Model has estimators_ attribute (ensemble detected)")

                print(f"\nModel attributes (first 20):")
                for i, attr in enumerate([a for a in dir(shap_model) if not a.startswith('_')][:20]):
                    print(f"  {i+1}. {attr}")

                print("=" * 70)

                raise RuntimeError(
                    f"Failed to initialize SHAP TreeExplainer. "
                    f"Final unwrapped model type: {type(shap_model).__name__} "
                    f"from module: {type(shap_model).__module__}. "
                    f"Original error: {e}"
                ) from e

        return self.shap_explainer

    def get_global_shap_importance(self, max_samples=50):
        explainer = self.initialize_shap(max_samples)
        sample_idx = np.random.choice(len(self.X_processed), size=min(max_samples, len(self.X_processed)), replace=False)
        X_sample = self.X_processed[sample_idx]
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = np.abs(shap_values).mean(axis=0)

        # CRITICAL: Ensure importance and feature_names have same length
        if len(importance) != len(self.feature_names):
            print(f"⚠️ WARNING: Length mismatch in SHAP importance!")
            print(f"  → importance array length: {len(importance)}")
            print(f"  → feature_names length: {len(self.feature_names)}")

            # Synchronize to shorter length
            min_len = min(len(importance), len(self.feature_names))
            print(f"  → Truncating both to {min_len} features")
            importance = importance[:min_len]
            feature_names_aligned = self.feature_names[:min_len]
        else:
            feature_names_aligned = self.feature_names

        return pd.DataFrame({'feature': feature_names_aligned, 'importance': importance}).sort_values('importance', ascending=False)

    def get_local_shap_explanation(self, device_index: int):
        explainer = self.initialize_shap()
        X_device = self.X_processed[device_index:device_index+1]
        shap_values = explainer.shap_values(X_device)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
        prediction = self.model.predict(X_device)[0]
        probability = self.model.predict_proba(X_device)[0, 1]
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        return {
            'device_index': device_index,
            'prediction': prediction,
            'probability': probability,
            'base_value': base_value,
            'shap_values': dict(zip(self.feature_names, shap_values)),
            'feature_values': dict(zip(self.feature_names, X_device[0]))
        }

    # ================================================================
    # PCA Feature Filtering — prevents label/ID leakage into PCA
    # ================================================================

    # Columns that must NEVER enter PCA: targets, quality labels,
    # identifiers, batch IDs, outcome flags, and categorical encodings
    # that represent labels rather than continuous process measurements.
    _PCA_EXCLUDE_EXACT = {
        # Target / outcome variables
        'Warranty_Claim', 'Claim_Type', 'Repair_Cost_USD',
        # Quality labels — binary Pass/Fail creates artificial separation
        'Internal_QC_Status',
        # Identifiers
        'Serial_Number', 'Device_UUID',
    }

    # Substrings: any feature whose name contains one of these tokens
    # is a batch ID, supplier ID, station ID, or firmware tag — not a
    # continuous process measurement.
    _PCA_EXCLUDE_PATTERNS = [
        'Batch_ID',          # PCB_Batch_ID, LCD_Batch_ID, Cable_Batch_ID
        'Supplier',          # *_Supplier columns (label-encoded categoricals)
        'Station_ID',        # Soldering_Station_ID — equipment identifier
        'Firmware_Version',  # software version tag, not a measurement
        'Thermal_Paste_Type',# categorical material label
        'Shift',             # categorical shift label (Morning/Afternoon/Night)
        'Housing_Case',      # supplier / categorical
    ]

    def _select_pca_features(self):
        """
        Select only valid continuous process features for PCA.

        Returns a boolean mask over self.feature_names / self.X_processed
        columns identifying which features are safe for PCA.

        Exclusion rules (applied in order):
          1. Exact name match against _PCA_EXCLUDE_EXACT
          2. Substring match against _PCA_EXCLUDE_PATTERNS
          3. Zero-variance columns (constant after preprocessing)

        Raises warnings if the surviving feature set is suspiciously small.
        """
        if self.feature_names is None or len(self.feature_names) == 0:
            raise ValueError("feature_names not available — cannot filter for PCA")

        n_total = len(self.feature_names)
        include_mask = np.ones(n_total, dtype=bool)
        excluded_report = []

        for idx, name in enumerate(self.feature_names):
            # --- rule 1: exact name match ---
            # Strip sklearn prefixes like "num__" / "cat__" for matching
            clean_name = name.split('__')[-1] if '__' in name else name
            if clean_name in self._PCA_EXCLUDE_EXACT:
                include_mask[idx] = False
                excluded_report.append((name, 'target / quality label / ID'))
                continue

            # --- rule 2: substring match ---
            matched_pattern = None
            for pattern in self._PCA_EXCLUDE_PATTERNS:
                if pattern.lower() in clean_name.lower():
                    matched_pattern = pattern
                    break
            if matched_pattern:
                include_mask[idx] = False
                excluded_report.append((name, f'matches pattern "{matched_pattern}"'))
                continue

        # --- rule 3: zero-variance (constant) columns ---
        X_subset = self.X_processed[:, include_mask]
        variances = np.var(X_subset, axis=0)
        surviving_names = [self.feature_names[i] for i in range(n_total) if include_mask[i]]
        zero_var_local = np.where(variances < 1e-10)[0]
        if len(zero_var_local) > 0:
            # Map local indices back to global
            global_indices = np.where(include_mask)[0]
            for local_idx in zero_var_local:
                global_idx = global_indices[local_idx]
                include_mask[global_idx] = False
                excluded_report.append((self.feature_names[global_idx], 'zero variance'))

        # --- report ---
        n_included = include_mask.sum()
        print("\n" + "=" * 60)
        print("PCA FEATURE AUDIT")
        print("=" * 60)
        print(f"Total preprocessed features : {n_total}")
        print(f"Excluded from PCA           : {n_total - n_included}")
        print(f"Included in PCA             : {n_included}")
        print("-" * 60)
        if excluded_report:
            print("EXCLUDED columns:")
            for col, reason in sorted(excluded_report, key=lambda x: x[0]):
                print(f"  ✗ {col:40s} → {reason}")
        print("-" * 60)
        included_names = [self.feature_names[i] for i in range(n_total) if include_mask[i]]
        print("INCLUDED columns (PCA feature set):")
        for col in sorted(included_names):
            print(f"  ✓ {col}")
        print("=" * 60 + "\n")

        # --- safety: warn if too few features survive ---
        if n_included < 3:
            import warnings
            warnings.warn(
                f"PCA feature set has only {n_included} features after filtering. "
                "Check _PCA_EXCLUDE lists — the filter may be too aggressive."
            )

        return include_mask

    def _validate_pca_input(self, X_pca_input, feature_names_pca):
        """
        Safety checks before running PCA.

        Raises ValueError if:
          - Any known target/label column survived filtering
          - Feature matrix has fewer than 2 columns
          - All features have zero variance (data collapsed)
        """
        # Check for target leakage
        target_names = {'Warranty_Claim', 'Internal_QC_Status', 'Claim_Type'}
        for name in feature_names_pca:
            clean = name.split('__')[-1] if '__' in name else name
            if clean in target_names:
                raise ValueError(
                    f"Target/label column '{name}' is still present in PCA input. "
                    "This will create artificial separation."
                )

        # Minimum dimensionality
        if X_pca_input.shape[1] < 2:
            raise ValueError(
                f"PCA requires at least 2 features but got {X_pca_input.shape[1]}."
            )

        # Variance collapse
        total_var = np.var(X_pca_input, axis=0).sum()
        if total_var < 1e-8:
            raise ValueError(
                "All PCA features have near-zero variance — "
                "data may be constant or incorrectly preprocessed."
            )

    # PCA Clustering with clean feature filtering
    def perform_pca_clustering(self, n_components=2, n_clusters=4):
        """
        Perform PCA clustering using only valid continuous process features.

        The method filters out targets, quality labels, identifiers,
        batch IDs, and label-encoded categoricals before PCA so the
        resulting components reflect real process variance — not
        encoding artifacts.

        Args:
            n_components: Number of PCA components
            n_clusters: Number of K-Means clusters

        Returns:
            Dictionary with pca_df, explained_variance, cluster_stats,
            high_risk_cluster, and feature_audit metadata.
        """
        # ============================================================
        # STEP 1: Filter features — only continuous process columns
        # ============================================================
        include_mask = self._select_pca_features()
        X_pca_input = self.X_processed[:, include_mask]
        feature_names_pca = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if include_mask[i]
        ]

        # ============================================================
        # STEP 2: Validate — catch leakage before it causes artifacts
        # ============================================================
        self._validate_pca_input(X_pca_input, feature_names_pca)

        # ============================================================
        # STEP 3: Re-scale the filtered subset (the original scaler
        # was fit on ALL columns; after dropping some, the remaining
        # values are already standardised so this is a no-op in most
        # cases, but we guard against numerical drift).
        # ============================================================
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X_pca_scaled = _scaler.fit_transform(X_pca_input)

        # ============================================================
        # STEP 4: PCA and K-Means clustering
        # ============================================================
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_model.fit_transform(X_pca_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)

        # Risk scores from the FULL feature set (model needs all features)
        probabilities = self.model.predict_proba(self.X_processed)[:, 1]

        # ============================================================
        # STEP 5: Build result DataFrame with forced index alignment
        # ============================================================
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=self.X.index)
        pca_df['cluster'] = cluster_labels
        pca_df['risk_score'] = probabilities
        pca_df['risk_pct'] = (probabilities * 100).round(1)

        # Attach metadata columns for hover/stats (from raw data, not PCA input)
        if 'Cable_Harness_Supplier' in self.X.columns:
            pca_df['Cable_Harness_Supplier'] = self.X['Cable_Harness_Supplier'].values
        if 'Soldering_Temp_Real_C' in self.X.columns:
            pca_df['Soldering_Temp_Real_C'] = self.X['Soldering_Temp_Real_C'].values

        pca_df['actual_failure'] = self.y.apply(
            lambda x: 1 if x == 'Yes' else 0
        ).values

        # Convenience aliases
        pca_df['supplier'] = pca_df.get('Cable_Harness_Supplier')
        pca_df['solder_temp'] = pca_df.get('Soldering_Temp_Real_C')

        # Reset to clean integer index
        pca_df = pca_df.reset_index(drop=True)

        # ============================================================
        # STEP 6: Verification output
        # ============================================================
        print("\n" + "=" * 60)
        print("--- PCA_DF VERIFICATION (CLEAN PIPELINE) ---")
        print("=" * 60)
        print(f"PCA input features : {len(feature_names_pca)}")
        print(f"Explained variance : PC1={self.pca_model.explained_variance_ratio_[0]:.3f}, "
              f"PC2={self.pca_model.explained_variance_ratio_[1]:.3f}")
        print(f"Total explained    : {sum(self.pca_model.explained_variance_ratio_):.3f}")
        print(f"Rows               : {len(pca_df)}")
        print("\nFirst 3 rows:")
        print(pca_df.head(3).to_string())
        print("=" * 60 + "\n")

        # ============================================================
        # STEP 7: Cluster statistics
        # ============================================================
        cluster_stats = []
        for cluster_id in range(n_clusters):
            mask = pca_df['cluster'] == cluster_id
            cluster_data = pca_df[mask]

            stats = {
                'cluster': cluster_id,
                'count': mask.sum(),
                'avg_risk': cluster_data['risk_score'].mean(),
                'failure_count': int(cluster_data['actual_failure'].sum()),
                'failure_rate': cluster_data['actual_failure'].mean() * 100
            }

            if 'supplier' in pca_df.columns and pca_df['supplier'].notna().any():
                supplier_counts = cluster_data['supplier'].value_counts()
                stats['dominant_supplier'] = (
                    supplier_counts.index[0] if len(supplier_counts) > 0 else 'N/A'
                )
                stats['dominant_supplier_pct'] = (
                    (supplier_counts.iloc[0] / len(cluster_data) * 100)
                    if len(cluster_data) > 0 else 0
                )
                cables_x_count = (cluster_data['supplier'] == 'Cables-X').sum()
                stats['cables_x_count'] = int(cables_x_count)
                stats['cables_x_pct'] = (
                    (cables_x_count / len(cluster_data) * 100)
                    if len(cluster_data) > 0 else 0
                )

            if 'solder_temp' in pca_df.columns and pca_df['solder_temp'].notna().any():
                stats['avg_solder_temp'] = cluster_data['solder_temp'].mean()

            stats['high_risk_count'] = int((cluster_data['risk_score'] > 0.5).sum())
            cluster_stats.append(stats)

        # Identify high-risk cluster
        high_risk_cluster = max(cluster_stats, key=lambda x: x['avg_risk'])
        high_risk_mask = pca_df['cluster'] == high_risk_cluster['cluster']
        high_risk_centroid = {
            'PC1': pca_df[high_risk_mask]['PC1'].mean(),
            'PC2': pca_df[high_risk_mask]['PC2'].mean()
        }

        return {
            'pca_df': pca_df,
            'explained_variance': self.pca_model.explained_variance_ratio_,
            'cluster_stats': cluster_stats,
            'high_risk_cluster': high_risk_cluster,
            'high_risk_centroid': high_risk_centroid,
            'kmeans_centroids': kmeans.cluster_centers_,
            # Audit metadata so callers can inspect what was filtered
            'pca_feature_names': feature_names_pca,
            'n_features_excluded': int((~include_mask).sum()),
            'n_features_included': int(include_mask.sum()),
        }

    def train_random_forest_comparison(self):
        if self.rf_model is None:
            y_encoded = (self.y == 'Yes').astype(int)
            self.rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
            self.rf_model.fit(self.X_processed, y_encoded)
        return self.rf_model

    def compare_models(self):
        rf_model = self.train_random_forest_comparison()
        y_encoded = (self.y == 'Yes').astype(int)

        # Get predictions
        xgb_pred = self.model.predict(self.X_processed)
        rf_pred = rf_model.predict(self.X_processed)

        # Get probabilities for ROC-AUC
        xgb_proba = self.model.predict_proba(self.X_processed)[:, 1]
        rf_proba = rf_model.predict_proba(self.X_processed)[:, 1]

        metrics = {
            'Model': ['XGBoost', 'Random Forest'],
            'F1 Score': [f1_score(y_encoded, xgb_pred), f1_score(y_encoded, rf_pred)],
            'Precision': [precision_score(y_encoded, xgb_pred), precision_score(y_encoded, rf_pred)],
            'Recall': [recall_score(y_encoded, xgb_pred), recall_score(y_encoded, rf_pred)],
            'Accuracy': [accuracy_score(y_encoded, xgb_pred), accuracy_score(y_encoded, rf_pred)],
            'ROC-AUC': [roc_auc_score(y_encoded, xgb_proba), roc_auc_score(y_encoded, rf_proba)]
        }
        return pd.DataFrame(metrics)

    def create_distribution_violin(self, feature_indices=None, n_features=6):
        """
        Create violin plot showing distribution of top N most important features.

        This method uses SHAP values to determine feature importance, then creates
        violin plots showing the distribution of these features across the dataset,
        colored by the target variable (warranty claim status).

        Args:
            feature_indices: Optional list of specific feature indices to plot.
                           If None, uses top N features by SHAP importance.
            n_features: Number of top features to display (default: 6)
                       Only used if feature_indices is None.

        Returns:
            plotly.graph_objects.Figure: Interactive violin plot with plotly_dark theme
        """
        try:
            # Ensure data is loaded
            if self.X_processed is None or self.y is None:
                raise ValueError("Data not loaded. Cannot create violin plot.")

            print(f"✓ Starting violin plot creation...")
            print(f"  self.y shape: {len(self.y)}")
            print(f"  self.X_processed shape: {self.X_processed.shape}")

            # Step 1: Explicitly truncate to minimum shared length FIRST
            min_len = min(len(self.y), self.X_processed.shape[0])
            print(f"  → Truncating to minimum shared length: {min_len}")

            # Step 2: Create clean truncated arrays
            y_clean = np.array([1 if str(val).strip().lower() == 'yes' else 0 for val in self.y[:min_len]])
            X_clean = self.X_processed[:min_len]

            print(f"  → y_clean shape: {len(y_clean)}")
            print(f"  → X_clean shape: {X_clean.shape}")

            # Verify they match
            assert len(y_clean) == X_clean.shape[0], f"Length mismatch after truncation: {len(y_clean)} != {X_clean.shape[0]}"
            print(f"  ✓ Arrays aligned successfully")

            # Use these cleaned arrays for the rest of the method
            y_arr = y_clean
            X_data = X_clean

            # Determine which features to plot
            if feature_indices is None:
                # Get global SHAP importance to identify top features
                importance_df = self.get_global_shap_importance(max_samples=500)
                top_features = importance_df.head(n_features)['feature'].tolist()
                feature_indices = [self.feature_names.index(f) for f in top_features]
            else:
                top_features = [self.feature_names[i] for i in feature_indices]
                n_features = len(feature_indices)

            # Create subplots for each feature
            fig = make_subplots(
                rows=1,
                cols=n_features,
                subplot_titles=top_features,
                horizontal_spacing=0.05
            )

            # Colors: #10B981 for healthy (0), #EF4444 for failed (1)
            colors = {0: '#10B981', 1: '#EF4444'}
            labels = {0: 'Healthy', 1: 'Failed'}

            # Step 3: Update plotting loop with X_data and y_arr
            for col_idx, (feat_idx, feature_name) in enumerate(zip(feature_indices, top_features), start=1):
                # Step 4: Add try/except for individual feature plotting
                try:
                    feature_values = X_data[:, feat_idx]

                    # CRITICAL: Verify lengths are exactly the same
                    if len(feature_values) != len(y_arr):
                        print(f"⚠ Skipping feature '{feature_name}': feature_values={len(feature_values)}, y_arr={len(y_arr)}")
                        continue

                    # Verify both are numpy arrays for safe masking
                    feature_values = np.array(feature_values)
                    y_arr_verified = np.array(y_arr)

                    # Create violin traces for each class
                    for class_val in [0, 1]:
                        # Create boolean mask - ensure it's same length as feature_values
                        mask = (y_arr_verified == class_val)

                        # Triple verification before masking
                        assert len(mask) == len(feature_values), f"Mask length {len(mask)} != feature length {len(feature_values)}"
                        assert len(mask) == len(y_arr_verified), f"Mask length {len(mask)} != y_arr length {len(y_arr_verified)}"

                        # Apply mask to get filtered values
                        filtered_values = feature_values[mask]

                        # Ensure we have data to plot
                        if len(filtered_values) == 0:
                            print(f"⚠ No data for feature '{feature_name}', class {class_val}")
                            continue

                        fig.add_trace(
                            go.Violin(
                                y=filtered_values,
                                name=labels[class_val],
                                legendgroup=labels[class_val],
                                scalegroup=labels[class_val],
                                line_color=colors[class_val],
                                fillcolor=colors[class_val],
                                opacity=0.6,
                                showlegend=(col_idx == 1),  # Show legend only for first subplot
                                box_visible=True,
                                meanline_visible=True
                            ),
                            row=1,
                            col=col_idx
                        )

                except AssertionError as ae:
                    print(f"⚠ Assertion failed for feature '{feature_name}': {ae}")
                    continue
                except Exception as e:
                    print(f"⚠ Error plotting feature '{feature_name}' (index {feat_idx}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next feature instead of crashing
                    continue

            # Apply plotly_dark template and styling
            fig.update_layout(
                template='plotly_dark',
                title=dict(
                    text=f'Feature Distributions: Top {n_features} Most Important Features',
                    font=dict(size=20, color='white')
                ),
                violinmode='group',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color='white')
                ),
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0.2)',
                font=dict(color='white', family='Inter, sans-serif')
            )

            # Update y-axes labels
            fig.update_yaxes(title_text="Feature Value", gridcolor='rgba(255,255,255,0.1)')
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')

            print(f"✓ Violin plot created successfully, returning figure")
            return fig

        except Exception as e:
            # Return error message as a simple plot if visualization fails
            print(f"Error creating violin plot: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color='red')
            )
            fig.update_layout(template='plotly_dark')
            return fig

    def create_shap_waterfall_data(self, device_index: int, top_n=10):
        """
        Create waterfall chart data for local SHAP explanation.

        Args:
            device_index: Index of the device to explain
            top_n: Number of top features to show (default: 10)

        Returns:
            tuple: (waterfall_df, explanation_dict)
        """
        explanation = self.get_local_shap_explanation(device_index)

        # Sort SHAP values by absolute magnitude
        shap_items = sorted(
            explanation['shap_values'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        waterfall_df = pd.DataFrame({
            'feature': [item[0] for item in shap_items],
            'shap_value': [item[1] for item in shap_items]
        })

        return waterfall_df, explanation

    def create_pca_scatter(self, pca_results, mode='2d'):
        """
        Create scatter plot from PCA results coloured by continuous risk score.

        Previous version used a binary Risk_Category derived from actual_failure
        (the target label).  That caused points to split into two discrete bands
        — an artifact of colouring by the target, not of real process variance.

        This version colours by the model's continuous risk_score so the
        visualisation reflects predicted risk gradients across PCA space.
        High-risk points (risk > 0.5) are drawn on top with larger markers.

        Args:
            pca_results: Dictionary from perform_pca_clustering().
            mode: '2d' (only 2d supported currently).

        Returns:
            plotly.graph_objects.Figure
        """
        pca_df = pca_results['pca_df'].copy()
        high_risk_cluster = pca_results.get('high_risk_cluster', {})
        high_risk_centroid = pca_results.get('high_risk_centroid', {})

        # ============================================================
        # 1. RISK CATEGORY — based on model risk score, NOT the target
        # ============================================================
        risk_threshold = 0.5
        pca_df['Risk_Category'] = np.where(
            pca_df['risk_score'] > risk_threshold,
            "High Risk",
            "Normal"
        )

        # Sort so high-risk points are drawn on top
        pca_df = pca_df.sort_values('risk_score').reset_index(drop=True)

        # ============================================================
        # 2. BUILD FIGURE — two traces for size/opacity control
        # ============================================================
        fig = go.Figure()

        trace_configs = [
            ("Normal",    "#1F77B4", 6,  0.5),
            ("High Risk", "#FF0000", 10, 0.9),
        ]

        for category, colour, size, opacity in trace_configs:
            mask = pca_df['Risk_Category'] == category
            subset = pca_df[mask]
            if len(subset) == 0:
                continue

            custom_data_cols = [
                subset['risk_pct'].values,
                subset['cluster'].values,
            ]
            hover = (
                "<b>Risk:</b> %{customdata[0]:.1f}%<br>"
                "<b>Cluster:</b> %{customdata[1]}<br>"
            )

            if 'supplier' in subset.columns and subset['supplier'].notna().any():
                custom_data_cols.append(subset['supplier'].values)
                hover += "<b>Supplier:</b> %{customdata[2]}<br>"

            if 'solder_temp' in subset.columns and subset['solder_temp'].notna().any():
                custom_data_cols.append(subset['solder_temp'].values)
                idx = len(custom_data_cols) - 1
                hover += f"<b>Solder Temp:</b> %{{customdata[{idx}]:.1f}} C<br>"

            hover += "<extra></extra>"
            custom_data = np.column_stack(custom_data_cols)

            fig.add_trace(go.Scatter(
                x=subset['PC1'],
                y=subset['PC2'],
                mode='markers',
                marker=dict(
                    size=size,
                    color=colour,
                    opacity=opacity,
                    line=dict(width=0.5, color='white') if category == "High Risk" else dict(width=0),
                ),
                customdata=custom_data,
                hovertemplate=hover,
                name=category,
                legendgroup=category,
            ))

        # ============================================================
        # 3. ROOT CAUSE ANNOTATION
        # ============================================================
        if high_risk_centroid and high_risk_cluster:
            fig.add_annotation(
                x=high_risk_centroid['PC1'],
                y=high_risk_centroid['PC2'],
                text="HIGH RISK CLUSTER",
                showarrow=True,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=3,
                arrowcolor='#FF0000',
                ax=100,
                ay=-80,
                font=dict(size=14, color='white', family='Inter'),
                bgcolor='rgba(255, 0, 0, 0.85)',
                bordercolor='#FFFFFF',
                borderwidth=2,
                borderpad=6,
            )

        # ============================================================
        # 4. LAYOUT — axes labelled with explained variance
        # ============================================================
        explained_var = pca_results.get('explained_variance', [0, 0])
        n_feat = pca_results.get('n_features_included', '?')
        high_count = (pca_df['Risk_Category'] == "High Risk").sum()
        normal_count = (pca_df['Risk_Category'] == "Normal").sum()

        fig.update_layout(
            title=dict(
                text=(
                    f'PCA Risk Segmentation ({n_feat} process features) | '
                    f'{high_count} High Risk vs {normal_count} Normal'
                ),
                font=dict(size=16, color='white'),
            ),
            xaxis_title=f'PC1 ({explained_var[0]:.1%} variance)',
            yaxis_title=f'PC2 ({explained_var[1]:.1%} variance)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='white', family='Inter'),
            height=600,
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="right", x=0.99,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white', borderwidth=1,
                font=dict(size=13),
            ),
        )

        return fig

    def get_feature_importance_comparison(self, top_n=10):
        """
        Compare feature importance between XGBoost and Random Forest.

        Args:
            top_n: Number of top features to compare

        Returns:
            pd.DataFrame: Comparison of feature importance
        """
        # Get XGBoost SHAP importance
        xgb_importance = self.get_global_shap_importance(max_samples=500)

        # Get Random Forest feature importance
        rf_model = self.train_random_forest_comparison()
        rf_importance_vals = rf_model.feature_importances_

        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_importance_vals
        }).sort_values('importance', ascending=False)

        # Merge and get top N
        xgb_top = xgb_importance.head(top_n)[['feature', 'importance']].rename(
            columns={'importance': 'XGBoost'}
        )
        rf_top = rf_importance.head(top_n)[['feature', 'importance']].rename(
            columns={'importance': 'Random Forest'}
        )

        # Get unique features from both
        all_features = list(set(xgb_top['feature'].tolist() + rf_top['feature'].tolist()))

        comparison = pd.DataFrame({'feature': all_features})
        comparison = comparison.merge(xgb_top, on='feature', how='left')
        comparison = comparison.merge(rf_top, on='feature', how='left')
        comparison = comparison.fillna(0)

        return comparison.head(top_n)

    def create_correlation_heatmap(self, top_n=15):
        """
        Create correlation heatmap for top N most important features.

        Args:
            top_n: Number of top features to include

        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        print(f"✓ Starting correlation heatmap creation...")
        print(f"  X_processed shape: {self.X_processed.shape}")

        # Get top features by SHAP importance
        importance_df = self.get_global_shap_importance(max_samples=500)
        top_features = importance_df.head(top_n)['feature'].tolist()
        print(f"  Top {top_n} features selected: {top_features[:3]}...")

        # Get feature indices with safety check
        feature_indices = []
        valid_top_features = []
        for f in top_features:
            try:
                idx = self.feature_names.index(f)
                feature_indices.append(idx)
                valid_top_features.append(f)
            except ValueError:
                print(f"⚠ Feature '{f}' not found in feature_names, skipping")
                continue

        if len(feature_indices) == 0:
            raise ValueError("No valid features found for correlation heatmap")

        print(f"  Valid feature indices: {len(feature_indices)}")

        # Create correlation matrix
        feature_data = self.X_processed[:, feature_indices]
        corr_matrix = np.corrcoef(feature_data.T)
        print(f"  Correlation matrix shape: {corr_matrix.shape}")

        # Create heatmap using valid features only
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=valid_top_features,
            y=valid_top_features,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=f'Feature Correlation Heatmap (Top {len(valid_top_features)} Features)',
            xaxis_title='Feature',
            yaxis_title='Feature',
            height=600,
            width=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter')
        )

        print(f"✓ Correlation heatmap created successfully, returning figure")
        return fig

    def simulate_prediction(self, feature_values: dict):
        """
        Simulate prediction with custom feature values.

        Args:
            feature_values: Dictionary mapping feature names to values

        Returns:
            dict: Prediction results with probability and SHAP values
        """
        # Create feature vector
        X_sim = np.zeros((1, len(self.feature_names)))

        # Fill in provided values
        for feature, value in feature_values.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                X_sim[0, idx] = value

        # Fill in mean values for unspecified features
        for i, feature in enumerate(self.feature_names):
            if feature not in feature_values:
                X_sim[0, i] = self.X_processed[:, i].mean()

        # Make prediction
        probability = self.model.predict_proba(X_sim)[0, 1]
        prediction = self.model.predict(X_sim)[0]

        # Get SHAP values
        explainer = self.initialize_shap()
        shap_values = explainer.shap_values(X_sim)

        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]

        return {
            'prediction': prediction,
            'probability': float(probability),
            'shap_values': dict(zip(self.feature_names, shap_values)),
            'feature_values': dict(zip(self.feature_names, X_sim[0]))
        }

# ============================================================================
# SOLDER TIME × TEMPERATURE — STANDALONE ANALYTICS HELPERS
# ============================================================================
# These functions operate on a raw DataFrame and are called from app.py to
# power the four additional visualisations in the "Solder Temperature" tab:
#   A) Decision-boundary overlay
#   B) Density heatmap
#   C) Process-regime clustering
#   D) Optimal time-threshold alert


def _ensure_binary_target(df, target_col='Warranty_Claim'):
    """Convert target to 0/1 int, handling 'Yes'/'No' or already-numeric."""
    s = df[target_col]
    if s.dtype == object:
        return (s.str.strip().str.lower() == 'yes').astype(int)
    return s.astype(int)


def fit_2d_boundary(df, x_col, y_col, target_col='Warranty_Claim'):
    """
    A) Fit a LogisticRegression on two process columns to predict warranty risk.

    Returns (model, scaler) fitted on non-NaN rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as _SS

    work = df[[x_col, y_col, target_col]].dropna().copy()
    y = _ensure_binary_target(work, target_col)
    X = work[[x_col, y_col]].values

    scaler = _SS()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    model.fit(X_scaled, y)
    return model, scaler


def predict_grid(model, scaler, x_range, y_range, resolution=200):
    """
    A) Evaluate the fitted boundary model over a mesh.

    Returns (xx, yy, proba) where proba is P(failure) on the grid.
    """
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], resolution),
        np.linspace(y_range[0], y_range[1], resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    proba = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    return xx, yy, proba


def compute_density_grid(df, x_col, y_col, bins=40):
    """
    B) Fast 2-D histogram density over (x, y).

    Returns (x_edges, y_edges, density) where density is row-major counts.
    NaN rows are dropped before binning.
    """
    work = df[[x_col, y_col]].dropna()
    density, x_edges, y_edges = np.histogram2d(
        work[x_col].values, work[y_col].values, bins=bins,
    )
    # Transpose so axes match conventional (x→cols, y→rows) for heatmap
    return x_edges, y_edges, density.T


def cluster_regimes(df, x_col, y_col, n_clusters=2, target_col='Warranty_Claim'):
    """
    C) Unsupervised KMeans on (x, y) to detect process regimes.

    Returns (df_out, summary_df) where df_out has a 'process_regime' column
    and summary_df has per-regime statistics.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as _SS

    work = df[[x_col, y_col, target_col]].dropna().copy()
    work['_target_bin'] = _ensure_binary_target(work, target_col)

    scaler = _SS()
    X_scaled = scaler.fit_transform(work[[x_col, y_col]].values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work['process_regime'] = km.fit_predict(X_scaled)

    # Per-regime summary
    summary = work.groupby('process_regime').agg(
        count=(x_col, 'size'),
        avg_time=(x_col, 'mean'),
        avg_temp=(y_col, 'mean'),
        warranty_rate=('_target_bin', 'mean'),
    ).reset_index()
    summary['warranty_rate'] = (summary['warranty_rate'] * 100).round(2)
    summary = summary.rename(columns={
        'avg_time': f'Avg {x_col}',
        'avg_temp': f'Avg {y_col}',
        'warranty_rate': 'Warranty Rate (%)',
    })

    return work, summary


def find_time_threshold(df, time_col, target_col='Warranty_Claim'):
    """
    D) Scan candidate thresholds on *time_col* and pick the one that
    maximises Youden's J = TPR - FPR (balances sensitivity & specificity).

    Returns (best_threshold, stats_dict).
    """
    from sklearn.metrics import f1_score as _f1

    work = df[[time_col, target_col]].dropna().copy()
    y = _ensure_binary_target(work, target_col)
    x = work[time_col].values

    # Scan a reasonable set of candidate thresholds (quantiles)
    candidates = np.unique(np.percentile(x, np.arange(5, 96, 1)))

    best_j, best_thr = -1.0, float(np.median(x))
    for thr in candidates:
        pred = (x >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = thr

    # Compute final stats at the chosen threshold
    above = y[x >= best_thr]
    below = y[x < best_thr]
    rate_above = above.mean() * 100 if len(above) > 0 else 0.0
    rate_below = below.mean() * 100 if len(below) > 0 else 0.0
    pred_best = (x >= best_thr).astype(int)
    f1_val = _f1(y, pred_best, zero_division=0)

    stats = {
        'threshold': round(float(best_thr), 3),
        'warranty_rate_above': round(rate_above, 2),
        'warranty_rate_below': round(rate_below, 2),
        'count_above': int((x >= best_thr).sum()),
        'count_below': int((x < best_thr).sum()),
        'f1_at_threshold': round(float(f1_val), 3),
        'youden_j': round(float(best_j), 3),
        'uplift': round(rate_above - rate_below, 2),
    }
    return best_thr, stats


if __name__ == "__main__":
    """
    Test script for Analytics Engine.
    This code only executes when running this file directly,
    NOT when imported by other modules.
    """
    print("=" * 60)
    print("Analytics Engine - Initialization Test")
    print("=" * 60)

    try:
        # Step 1: Initialize the engine
        print("\n1. Initializing Analytics Engine...")
        engine = AnalyticsEngine()
        print("✓ Analytics Engine successfully initialized")

        # Step 2: Display model attributes
        print("\n2. Inspecting model structure...")
        print(f"   Model type: {type(engine.model).__name__}")
        print(f"   Model attributes (first 10): {dir(engine.model)[:10]}")

        # Step 3: Initialize SHAP
        print("\n3. Testing SHAP initialization...")
        explainer = engine.initialize_shap(max_samples=100)
        print("✓ SHAP initialization successful!")

        # Step 4: Test global importance
        print("\n4. Computing global SHAP importance...")
        importance = engine.get_global_shap_importance(max_samples=100)
        print("✓ Top 5 most important features:")
        print(importance.head())

        print("\n" + "=" * 60)
        print("All tests passed! Analytics Engine is ready.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
