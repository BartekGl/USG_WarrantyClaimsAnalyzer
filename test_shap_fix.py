"""
Quick test to verify SHAP initialization works after XGBoost base_score fix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import custom classes needed for unpickling
try:
    from ml_core import LabelEncoderPipeline
except ImportError:
    print("⚠ Warning: ml_core.py not found - some functionality may be limited")
    pass

def test_shap_initialization():
    """Test that SHAP explainer can be initialized without errors."""
    print("=" * 60)
    print("Testing SHAP Initialization Fix")
    print("=" * 60)

    try:
        # Import the analytics engine
        print("\n1. Importing AnalyticsEngine...")
        from analytics_engine import AnalyticsEngine
        print("   ✓ Import successful")

        # Initialize the engine (loads model and data)
        print("\n2. Initializing AnalyticsEngine...")
        engine = AnalyticsEngine()
        print("   ✓ Engine initialized")

        # Test SHAP initialization (this is where the error occurred)
        print("\n3. Initializing SHAP explainer...")
        explainer = engine.initialize_shap(max_samples=100)
        print("   ✓ SHAP explainer initialized successfully!")

        # Quick test: get global importance
        print("\n4. Testing SHAP feature importance calculation...")
        importance_df = engine.get_global_shap_importance(max_samples=100)
        print(f"   ✓ Calculated importance for {len(importance_df)} features")
        print(f"\n   Top 5 features:")
        print(importance_df.head(5).to_string(index=False))

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except FileNotFoundError as e:
        print(f"\n⚠ Warning: Required files not found")
        print(f"   {e}")
        print("\n   This test requires:")
        print("   - models/model.pkl")
        print("   - models/preprocessor.pkl")
        print("   - data/raw/USG_Data_cleared.csv")
        print("\n   Please train the model first or provide these files.")
        return False

    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_shap_initialization()
    sys.exit(0 if success else 1)
