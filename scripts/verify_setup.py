"""
Quick verification script to test if environment is set up correctly
Run this to verify everything is working before training
"""

import sys
from pathlib import Path

print("="*60)
print("USG FAILURE PREDICTION - ENVIRONMENT VERIFICATION")
print("="*60)

# Test 1: Python version
print("\n✓ Python Version:")
print(f"  {sys.version}")
if sys.version_info < (3, 10):
    print("  ⚠ WARNING: Python 3.10+ recommended")
else:
    print("  ✓ Version OK")

# Test 2: Required packages
print("\n✓ Checking Required Packages:")
required_packages = [
    'numpy',
    'pandas',
    'sklearn',
    'xgboost',
    'lightgbm',
    'shap',
    'optuna',
    'fastapi',
    'joblib',
    'matplotlib',
    'seaborn'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING!")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
    print("  Run: pip install -r requirements.txt")
else:
    print("\n✓ All required packages installed!")

# Test 3: Data file
print("\n✓ Checking Data File:")
data_file = Path('data/raw/USG_Data_cleared.csv')
if data_file.exists():
    print(f"  ✓ Found: {data_file}")
    import pandas as pd
    df = pd.read_csv(data_file)
    print(f"  ✓ Shape: {df.shape}")
    if 'Warranty_Claim' in df.columns:
        print(f"  ✓ Target column 'Warranty_Claim' found")
    else:
        print(f"  ✗ Target column 'Warranty_Claim' NOT found!")
else:
    print(f"  ✗ NOT FOUND: {data_file}")
    print("  Please place USG_Data_cleared.csv in data/raw/ directory")

# Test 4: Directory structure
print("\n✓ Checking Directory Structure:")
required_dirs = [
    'data/raw',
    'data/processed',
    'models',
    'reports/visualizations',
    'reports/metrics',
    'src',
    'scripts'
]

for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ {dir_path} - will be created during training")

# Test 5: Source files
print("\n✓ Checking Source Files:")
source_files = [
    'src/model.py',
    'src/preprocessing.py',
    'src/evaluation.py',
    'src/api.py',
    'scripts/train_model.py',
    'scripts/run_shap_analysis.py'
]

all_source_ok = True
for file_path in source_files:
    if Path(file_path).exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - MISSING!")
        all_source_ok = False

# Test 6: Can import source modules
print("\n✓ Testing Module Imports:")
sys.path.insert(0, str(Path('src')))

try:
    from model import USGFailurePredictionModel
    print("  ✓ model.py imports successfully")
except Exception as e:
    print(f"  ✗ model.py import failed: {e}")

try:
    from preprocessing import USGPreprocessingPipeline
    print("  ✓ preprocessing.py imports successfully")
except Exception as e:
    print(f"  ✗ preprocessing.py import failed: {e}")

try:
    from evaluation import ModelEvaluator
    print("  ✓ evaluation.py imports successfully")
except Exception as e:
    print(f"  ✗ evaluation.py import failed: {e}")

# Final verdict
print("\n" + "="*60)
if not missing_packages and data_file.exists() and all_source_ok:
    print("✓ READY TO TRAIN!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run training: scripts\\train.bat")
    print("  2. Generate SHAP: scripts\\shap.bat")
    print("  3. Start API: scripts\\start_api.bat")
else:
    print("⚠ SETUP INCOMPLETE")
    print("="*60)
    print("\nPlease fix the issues above before training.")
    if missing_packages:
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
    if not data_file.exists():
        print("\nAdd your data file:")
        print("  Place USG_Data_cleared.csv in data/raw/ directory")

print("\n" + "="*60)
