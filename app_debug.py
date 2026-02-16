"""
USG Failure Prediction - DEBUG MODE Dashboard
This version includes extensive debugging output to diagnose issues.

Run: streamlit run app_debug.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback

import shap
import streamlit.components.v1 as components
from ml_core import LabelEncoderPipeline, USGModelPipeline
from analytics_engine import AnalyticsEngine
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="USG Mission Control - DEBUG",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DEBUG HELPER ====================

def debug_status(message, status="info"):
    """Display debug status messages."""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    colors = {"info": "blue", "success": "green", "warning": "orange", "error": "red"}
    st.markdown(f":{colors[status]}[{icons[status]} **DEBUG:** {message}]")

def safe_call(func, *args, **kwargs):
    """Safely call a function and return result or error."""
    try:
        result = func(*args, **kwargs)
        debug_status(f"Successfully called {func.__name__}", "success")
        return result, None
    except AttributeError as e:
        debug_status(f"Method {func.__name__} does not exist: {e}", "error")
        return None, str(e)
    except Exception as e:
        debug_status(f"Error in {func.__name__}: {e}", "error")
        st.code(traceback.format_exc())
        return None, str(e)

# ==================== CUSTOM CSS ====================

def apply_custom_css():
    """Apply minimal styling for debug mode."""
    st.markdown("""
    <style>
        .debug-box {
            background: #f0f0f0;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            font-family: monospace;
        }
        .error-box {
            background: #ffe6e6;
            border: 2px solid #ff0000;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }
        .success-box {
            background: #e6ffe6;
            border: 2px solid #00aa00;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ==================== HEADER ====================

st.title("🐛 USG MISSION CONTROL - DEBUG MODE")
st.markdown("---")
debug_status("Starting debug mode dashboard initialization", "info")

# ==================== INITIALIZE ANALYTICS ENGINE ====================

st.header("1️⃣ Loading Analytics Engine")

@st.cache_resource
def load_analytics_engine():
    """Load analytics engine with debug output."""
    debug_status("Attempting to load AnalyticsEngine...", "info")
    try:
        engine = AnalyticsEngine()
        debug_status("AnalyticsEngine loaded successfully", "success")
        return engine, None
    except Exception as e:
        error_msg = f"Failed to load analytics engine: {str(e)}"
        debug_status(error_msg, "error")
        st.code(traceback.format_exc())
        return None, error_msg

engine, error = load_analytics_engine()

if error:
    st.error(f"❌ Cannot continue: {error}")
    st.stop()

# ==================== ENGINE INSPECTION ====================

st.header("2️⃣ Analytics Engine Inspection")

with st.expander("🔍 View Engine Attributes", expanded=True):
    debug_status("Inspecting engine attributes...", "info")

    # Check basic attributes
    checks = {
        "engine.model": hasattr(engine, 'model'),
        "engine.preprocessor": hasattr(engine, 'preprocessor'),
        "engine.X": hasattr(engine, 'X'),
        "engine.y": hasattr(engine, 'y'),
        "engine.X_processed": hasattr(engine, 'X_processed'),
        "engine.feature_names": hasattr(engine, 'feature_names'),
        "engine.shap_explainer": hasattr(engine, 'shap_explainer')
    }

    for attr, exists in checks.items():
        if exists:
            value = getattr(engine, attr.split('.')[1])
            if value is not None:
                debug_status(f"{attr} exists and is not None", "success")
                if hasattr(value, 'shape'):
                    st.write(f"  - Shape: {value.shape}")
                elif isinstance(value, (list, pd.Series)):
                    st.write(f"  - Length: {len(value)}")
            else:
                debug_status(f"{attr} exists but is None", "warning")
        else:
            debug_status(f"{attr} does not exist", "error")

    # List all available methods
    st.markdown("### Available Methods:")
    methods = [m for m in dir(engine) if callable(getattr(engine, m)) and not m.startswith('_')]
    st.write(methods)

# ==================== DATA SUMMARY ====================

st.header("3️⃣ Data Summary")

if hasattr(engine, 'y') and engine.y is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        total = len(engine.y)
        st.metric("Total Devices", total)
        debug_status(f"Found {total} devices", "success")

    with col2:
        failures = (engine.y == 'Yes').sum()
        st.metric("Failures", failures)
        debug_status(f"Found {failures} failures", "success")

    with col3:
        if hasattr(engine, 'X_processed') and engine.X_processed is not None:
            n_features = engine.X_processed.shape[1]
            st.metric("Features", n_features)
            debug_status(f"Found {n_features} features", "success")
        else:
            debug_status("X_processed not available", "warning")
else:
    debug_status("Target variable (y) not available", "error")

st.markdown("---")

# ==================== TEST EACH METHOD ====================

st.header("4️⃣ Method Testing")

# Test 1: create_distribution_violin
st.subheader("Test: create_distribution_violin")
if hasattr(engine, 'create_distribution_violin'):
    debug_status("Method exists", "success")
    try:
        with st.spinner("Testing create_distribution_violin..."):
            fig = engine.create_distribution_violin(n_features=6)
            debug_status("Method executed successfully", "success")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        debug_status(f"Method failed: {e}", "error")
        st.code(traceback.format_exc())
else:
    debug_status("Method does not exist", "error")

st.markdown("---")

# Test 2: get_global_shap_importance
st.subheader("Test: get_global_shap_importance")
if hasattr(engine, 'get_global_shap_importance'):
    debug_status("Method exists", "success")
    try:
        with st.spinner("Testing get_global_shap_importance..."):
            importance_df = engine.get_global_shap_importance(max_samples=100)
            debug_status("Method executed successfully", "success")
            st.write(f"Returned shape: {importance_df.shape}")
            st.dataframe(importance_df.head(10))

            # Create a simple bar chart
            fig = px.bar(
                importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Features'
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        debug_status(f"Method failed: {e}", "error")
        st.code(traceback.format_exc())
else:
    debug_status("Method does not exist", "error")

st.markdown("---")

# Test 3: get_local_shap_explanation
st.subheader("Test: get_local_shap_explanation")
if hasattr(engine, 'get_local_shap_explanation'):
    debug_status("Method exists", "success")
    device_idx = st.slider("Select device index", 0, len(engine.y)-1, 0)
    try:
        with st.spinner("Testing get_local_shap_explanation..."):
            explanation = engine.get_local_shap_explanation(device_idx)
            debug_status("Method executed successfully", "success")
            st.json(explanation)
    except Exception as e:
        debug_status(f"Method failed: {e}", "error")
        st.code(traceback.format_exc())
else:
    debug_status("Method does not exist", "error")

st.markdown("---")

# Test 4: perform_pca_clustering
st.subheader("Test: perform_pca_clustering")
if hasattr(engine, 'perform_pca_clustering'):
    debug_status("Method exists", "success")
    try:
        with st.spinner("Testing perform_pca_clustering..."):
            pca_results = engine.perform_pca_clustering(n_components=2, n_clusters=4)
            debug_status("Method executed successfully", "success")
            st.write("PCA Results keys:", list(pca_results.keys()))

            # Try to visualize
            if 'pca_df' in pca_results:
                pca_df = pca_results['pca_df']
                st.write(f"PCA DataFrame shape: {pca_df.shape}")
                st.dataframe(pca_df.head())

                # Create simple scatter
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='cluster',
                    title='PCA Clustering Results'
                )
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        debug_status(f"Method failed: {e}", "error")
        st.code(traceback.format_exc())
else:
    debug_status("Method does not exist", "error")

st.markdown("---")

# Test 5: compare_models
st.subheader("Test: compare_models")
if hasattr(engine, 'compare_models'):
    debug_status("Method exists", "success")
    try:
        with st.spinner("Testing compare_models..."):
            comparison_df = engine.compare_models()
            debug_status("Method executed successfully", "success")
            st.dataframe(comparison_df)
    except Exception as e:
        debug_status(f"Method failed: {e}", "error")
        st.code(traceback.format_exc())
else:
    debug_status("Method does not exist", "error")

st.markdown("---")

# Test 6: Check for missing methods
st.subheader("Missing Methods Check")

missing_methods = [
    'create_shap_waterfall_data',
    'create_pca_scatter',
    'get_feature_importance_comparison',
    'create_correlation_heatmap',
    'simulate_prediction'
]

for method in missing_methods:
    if hasattr(engine, method):
        debug_status(f"Method {method} exists", "success")
    else:
        debug_status(f"Method {method} is MISSING - app.py will fail when calling it", "error")

# ==================== SUMMARY ====================

st.header("📋 Debug Summary")

st.markdown("""
<div class='debug-box'>
<h3>Key Findings:</h3>
<ol>
<li>Check if engine loaded successfully ✅ or ❌</li>
<li>Check if all data attributes are present (X, y, X_processed, etc.)</li>
<li>Check which methods work and which fail</li>
<li>Identify missing methods that app.py is trying to call</li>
</ol>

<h3>Common Issues:</h3>
<ul>
<li>If SHAP methods fail: Check sniper patch is working</li>
<li>If methods are missing: They need to be added to analytics_engine.py</li>
<li>If data is None: Check data loading in _load_data()</li>
<li>If caching issues: Clear cache with Ctrl+Shift+R or st.cache_resource.clear()</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐛 Debug Mode Dashboard | Use this to diagnose issues before running the main app</p>
</div>
""", unsafe_allow_html=True)
