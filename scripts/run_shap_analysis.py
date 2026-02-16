"""
SHAP Analysis Script
Generates interpretability visualizations after model training
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import train_test_split

print("="*60)
print("USG FAILURE PREDICTION - SHAP ANALYSIS")
print("="*60)

# Check if model exists
model_path = Path('models/model.pkl')
if not model_path.exists():
    print("\n❌ Error: Model not found!")
    print("Please run training first: python scripts/train_model.py")
    sys.exit(1)

# Load model and data
print("\nLoading model and data...")
model = joblib.load('models/model.pkl')
X = pd.read_csv('data/processed/X_processed.csv')
y = pd.read_csv('data/processed/y_target.csv').squeeze()

# Split data (same split as training)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Loaded model and test data: {X_test.shape}")

# Create SHAP explainer
print("\nInitializing SHAP explainer...")
print("This may take 1-2 minutes...")

# Get base model from calibrated wrapper
if hasattr(model, 'model'):
    if hasattr(model.model, 'calibrated_model'):
        base_estimator = model.model.calibrated_model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = model.model
else:
    base_estimator = model

# Create explainer
explainer = shap.TreeExplainer(base_estimator)

# Calculate SHAP values
print("Calculating SHAP values for test set...")
shap_values = explainer.shap_values(X_test)

# For binary classification, get positive class
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]
else:
    shap_values_positive = shap_values

print(f"✓ SHAP values computed: {shap_values_positive.shape}")

# Save explainer
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("✓ SHAP explainer saved to models/shap_explainer.pkl")

# Create visualizations directory
Path('reports/visualizations').mkdir(parents=True, exist_ok=True)

print("\nGenerating visualizations...")

# 1. SHAP Summary Plot (beeswarm)
print("  - Creating summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_positive, X_test, max_display=20, show=False)
plt.title('SHAP Summary Plot - Top 20 Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/visualizations/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. SHAP Bar Plot (mean absolute impact)
print("  - Creating bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_positive, X_test, plot_type="bar", max_display=20, show=False)
plt.title('Feature Importance - Mean |SHAP Value|', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/visualizations/shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature importance rankings
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values_positive).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print("="*60)
for idx, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:40s}: {row['importance']:.4f}")
print("="*60)

# Save feature importance
feature_importance.to_csv('reports/metrics/shap_feature_importance.csv', index=False)

# 4. Waterfall plots for sample predictions
print("  - Creating waterfall plots...")

# Find example cases
failure_idx = y_test[y_test == 'Yes'].index[0]
no_failure_idx = y_test[y_test == 'No'].index[0]

failure_pos = X_test.index.get_loc(failure_idx)
no_failure_pos = X_test.index.get_loc(no_failure_idx)

base_value = explainer.expected_value
if isinstance(base_value, list):
    base_value = base_value[1]

# Waterfall for failure case
explanation_failure = shap.Explanation(
    values=shap_values_positive[failure_pos],
    base_values=base_value,
    data=X_test.iloc[failure_pos].values,
    feature_names=X_test.columns.tolist()
)

plt.figure(figsize=(10, 8))
shap.plots.waterfall(explanation_failure, max_display=20, show=False)
plt.title('SHAP Waterfall - Failure Case', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/visualizations/shap_waterfall_failure.png', dpi=300, bbox_inches='tight')
plt.close()

# Waterfall for no-failure case
explanation_no_failure = shap.Explanation(
    values=shap_values_positive[no_failure_pos],
    base_values=base_value,
    data=X_test.iloc[no_failure_pos].values,
    feature_names=X_test.columns.tolist()
)

plt.figure(figsize=(10, 8))
shap.plots.waterfall(explanation_no_failure, max_display=20, show=False)
plt.title('SHAP Waterfall - No Failure Case', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/visualizations/shap_waterfall_no_failure.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ All visualizations saved to reports/visualizations/")

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  - reports/visualizations/shap_summary_plot.png")
print("  - reports/visualizations/shap_bar_plot.png")
print("  - reports/visualizations/shap_waterfall_failure.png")
print("  - reports/visualizations/shap_waterfall_no_failure.png")
print("  - reports/metrics/shap_feature_importance.csv")
print("  - models/shap_explainer.pkl")
print("\nTop predictive feature:", feature_importance.iloc[0]['feature'])
print("="*60)
