"""
Business Intelligence Module for Executive Quality Dashboard
Transforms technical ML outputs into Board-level business risk metrics.

This module provides:
1. Risk Scoring Engine with business category grouping
2. Segment-based analysis with High-Risk Cluster identification
3. What-If scenario engine for process optimization
4. Financial impact calculation (Cost of Quality)
5. Supplier scorecard generation
"""

import sys
from pathlib import Path

# Add project root to path for imports during unpickling
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))  # For ml_core.py in root
sys.path.insert(0, str(project_root / 'src'))  # For legacy src modules

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap
import pickle

# Import custom classes needed for unpickling
try:
    from ml_core import LabelEncoderPipeline
except ImportError:
    # Fallback: create placeholder class
    class LabelEncoderPipeline:
        """Placeholder for unpickling preprocessor.pkl"""
        pass

# Pickle compatibility: Handle __main__.LabelEncoderPipeline redirects
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
    """Load pickle file with __main__ compatibility"""
    with open(filepath, 'rb') as f:
        return CompatibilityUnpickler(f).load()


# ============================================================================
# BUSINESS CATEGORY MAPPING
# ============================================================================

# Business Pillars for Executive Reporting
BUSINESS_PILLARS = {
    "Assembly Quality": [
        'Soldering_Time_s',
        'Soldering_Temp_Set_C',
        'Soldering_Temp_Real_C',
        'Soldering_Tip_Age_Cycles',
        'Flux_Usage_mg',
        'Smoke_Extractor_Flow_m3h',
        'Screw_Torque_Nm',
        'Screw_Count_OK',
        'Glue_Dispense_Weight_mg',
        'Housing_Gap_Check_mm'
    ],

    "Supplier Reliability": [
        'PCB_Mainboard_Supplier',
        'PCB_Batch_ID',
        'LCD_Panel_Supplier',
        'LCD_Batch_ID',
        'Cable_Harness_Supplier',
        'Cable_Batch_ID',
        'Battery_Pack_Supplier',
        'Housing_Case_Supplier'
    ],

    "Software/System": [
        'Firmware_Version',
        'Boot_Time_s',
        'Standby_Current_mA',
        'Max_Load_Current_A'
    ],

    "Environmental Controls": [
        'Ambient_Temp_C',
        'Relative_Humidity',
        'Air_Pressure_hPa',
        'Particulate_PM2.5',
        'Particulate_PM10'
    ],

    "Component Testing": [
        'Battery_Charge_Rate_A',
        'Screen_Brightness_Nits',
        'Dead_Pixel_Count',
        'Touch_Response_ms',
        'WiFi_Signal_dBm',
        'Audio_Output_dB',
        'Probe_Connector_Impedance_Ohm'
    ]
}

# Component Category Mapping (Isolated Supplier Analysis)
COMPONENT_CATEGORIES = {
    'Cable_Harness_Supplier': 'Cables',
    'Battery_Pack_Supplier': 'Batteries',
    'PCB_Mainboard_Supplier': 'PCB Mainboards',
    'LCD_Panel_Supplier': 'LCD Panels',
    'Housing_Case_Supplier': 'Housing Cases'
}

# Legacy mapping for backward compatibility
BUSINESS_CATEGORIES = BUSINESS_PILLARS


@dataclass
class RiskScore:
    """Business risk score container"""
    category: str
    risk_score: float  # 0-100 scale
    contribution_pct: float  # % of total risk
    top_drivers: List[Tuple[str, float]]  # (feature_name, impact)
    recommendation: str


@dataclass
class FinancialImpact:
    """Cost of Quality metrics"""
    total_projected_loss_usd: float
    potential_savings_usd: float
    roi_improvement_pct: float
    avg_repair_cost_usd: float = 1200.0  # Industry standard for USG repair


@dataclass
class SupplierMetrics:
    """Supplier performance scorecard with component isolation"""
    supplier_name: str
    component_category: str  # "Cables", "Batteries", "PCB Mainboards", etc.
    failure_rate_pct: float
    units_affected: int
    total_units: int  # Total units from this supplier
    risk_level: str  # "Critical", "High", "Medium", "Low"
    financial_impact_usd: float
    vs_category_avg_pct: float  # Performance vs category average


@dataclass
class CEOBrief:
    """CEO Executive Brief with key financial metrics"""
    top_technical_cause: str
    top_supplier_financial_leak: str
    supplier_switch_savings_usd: float
    current_supplier: str
    recommended_supplier: str
    risk_reduction_pct: float
    one_time_investment_usd: float = 50000.0
    net_annual_savings_usd: float = 0.0
    roi_first_year_pct: float = 0.0


# ============================================================================
# BUSINESS RISK SCORING ENGINE
# ============================================================================

class BusinessRiskEngine:
    """
    Converts technical SHAP values into Board-level business risk drivers.

    Key capabilities:
    - Aggregate features into business categories
    - Identify high-risk clusters (Cluster 4)
    - Calculate financial impact
    - Generate supplier scorecards
    - Run what-if scenarios
    """

    def __init__(
        self,
        model_path: str = 'models/model.pkl',
        preprocessor_path: str = 'models/preprocessor.pkl',
        data_path: str = 'data/raw/USG_Data_cleared.csv',
        avg_repair_cost: float = 1200.0
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.data_path = Path(data_path)
        self.avg_repair_cost = avg_repair_cost

        # Load models
        self.model = None
        self.preprocessor = None
        self.shap_explainer = None
        self.feature_names = []

        # Business metrics
        self.risk_clusters = None
        self.optimal_soldering_time = None

    def load_models(self):
        """Load ML models and initialize SHAP"""
        print("📊 Loading models for business intelligence...")

        if self.model_path.exists():
            # Use compatibility unpickler to handle __main__ module redirects
            try:
                self.model = load_pickle_with_compat(self.model_path)
                print(f"✓ Model loaded (with compatibility)")
            except Exception as e:
                # Fallback to joblib.load if custom unpickler fails
                print(f"⚠ Compatibility unpickler failed: {e}, trying joblib.load...")
                self.model = joblib.load(self.model_path)
                print(f"✓ Model loaded")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if self.preprocessor_path.exists():
            # Use compatibility unpickler to handle __main__ module redirects
            try:
                self.preprocessor = load_pickle_with_compat(self.preprocessor_path)
                print(f"✓ Preprocessor loaded (with compatibility)")
            except Exception as e:
                # Fallback to joblib.load if custom unpickler fails
                print(f"⚠ Compatibility unpickler failed: {e}, trying joblib.load...")
                self.preprocessor = joblib.load(self.preprocessor_path)
                print(f"✓ Preprocessor loaded")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and preprocess data for analysis"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        print(f"✓ Data loaded: {len(df):,} devices")

        return df

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create manufacturing-focused interaction features.
        Must match the feature engineering done during model training.

        Args:
            X: Feature dataframe

        Returns:
            X with added interaction features
        """
        X = X.copy()

        # Critical interaction: Solder Temperature × Supplier
        if 'Solder_Temperature' in X.columns and 'Component_Supplier' in X.columns:
            X['Solder_Temperature_x_Supplier'] = (
                X['Solder_Temperature'].astype(str) + '_' +
                X['Component_Supplier'].astype(str)
            )

        return X

    def aggregate_shap_by_category(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> List[RiskScore]:
        """
        Aggregate technical SHAP values into business categories.

        Args:
            shap_values: SHAP values for a sample (shape: n_features)
            feature_names: List of feature names

        Returns:
            List of RiskScore objects, sorted by contribution
        """
        # Build reverse mapping: feature -> category
        feature_to_category = {}
        for category, features in BUSINESS_CATEGORIES.items():
            for feature in features:
                feature_to_category[feature] = category

        # Aggregate SHAP values by category
        category_scores = {}
        category_drivers = {}

        for feat_name, shap_val in zip(feature_names, shap_values):
            category = feature_to_category.get(feat_name, "Other")

            if category not in category_scores:
                category_scores[category] = 0.0
                category_drivers[category] = []

            category_scores[category] += abs(shap_val)
            category_drivers[category].append((feat_name, shap_val))

        # Calculate total risk
        total_risk = sum(category_scores.values())

        # Create RiskScore objects
        risk_scores = []
        for category, score in category_scores.items():
            contribution_pct = (score / total_risk * 100) if total_risk > 0 else 0

            # Sort drivers by absolute impact
            top_drivers = sorted(
                category_drivers[category],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]  # Top 3 drivers

            # Generate recommendation
            recommendation = self._generate_recommendation(category, top_drivers)

            risk_scores.append(RiskScore(
                category=category,
                risk_score=score * 100,  # Scale to 0-100
                contribution_pct=contribution_pct,
                top_drivers=top_drivers,
                recommendation=recommendation
            ))

        # Sort by contribution (highest first)
        risk_scores.sort(key=lambda x: x.contribution_pct, reverse=True)

        return risk_scores

    def _generate_recommendation(
        self,
        category: str,
        top_drivers: List[Tuple[str, float]]
    ) -> str:
        """Generate actionable business recommendation"""
        recommendations = {
            "Soldering Process Integrity":
                "Audit soldering stations and retrain operators. Target: 95% within spec.",
            "Supplier Quality Risk":
                "Initiate supplier audit program. Escalate high-risk vendors to procurement.",
            "Environmental/Factory Conditions":
                "Install real-time environmental monitoring. Set alerts for out-of-spec conditions.",
            "Software Stability":
                "Review firmware rollout process. Implement staged deployment strategy.",
            "Assembly Quality":
                "Recalibrate assembly equipment. Increase inline QC checks.",
            "Component Testing":
                "Review test procedures and equipment calibration. Update acceptance criteria."
        }
        return recommendations.get(category, "Review process parameters and conduct root cause analysis.")

    def identify_high_risk_clusters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform clustering analysis to identify high-risk segments.
        Returns cluster labels and statistics.

        Args:
            X: Feature matrix
            y: Target (Warranty_Claim: Yes/No)
            n_clusters: Number of clusters

        Returns:
            cluster_labels, cluster_stats dict
        """
        print(f"\n🔍 Identifying high-risk clusters...")

        # Preprocess data
        X_processed = self.preprocessor.transform(X)

        # Standardize for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate cluster statistics
        y_binary = (y == 'Yes').astype(int)
        cluster_stats = {}

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_size = mask.sum()
            failure_rate = y_binary[mask].mean()

            cluster_stats[cluster_id] = {
                'size': int(cluster_size),
                'failure_rate': float(failure_rate),
                'devices': int(cluster_size),
                'failures': int(y_binary[mask].sum())
            }

            print(f"  Cluster {cluster_id}: {cluster_size:,} devices, "
                  f"{failure_rate*100:.1f}% failure rate")

        # Identify high-risk cluster (highest failure rate)
        high_risk_id = max(cluster_stats.keys(), key=lambda k: cluster_stats[k]['failure_rate'])
        print(f"\n⚠️  HIGH-RISK CLUSTER: Cluster {high_risk_id} "
              f"({cluster_stats[high_risk_id]['failure_rate']*100:.1f}% failure rate)")

        self.risk_clusters = cluster_stats

        return cluster_labels, cluster_stats

    def calculate_financial_impact(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        scenario_pred: Optional[np.ndarray] = None
    ) -> FinancialImpact:
        """
        Calculate Cost of Quality metrics.

        Args:
            y_true: Actual warranty claims (Yes/No)
            y_pred: Current model predictions
            scenario_pred: Predictions after optimization (optional)

        Returns:
            FinancialImpact object
        """
        y_binary = (y_true == 'Yes').astype(int)

        # Current state
        actual_failures = y_binary.sum()
        total_projected_loss = actual_failures * self.avg_repair_cost

        # Potential savings (if scenario provided)
        if scenario_pred is not None:
            scenario_failures = scenario_pred.sum()
            failures_prevented = actual_failures - scenario_failures
            potential_savings = failures_prevented * self.avg_repair_cost
            roi_improvement = (failures_prevented / actual_failures * 100) if actual_failures > 0 else 0
        else:
            potential_savings = 0
            roi_improvement = 0

        return FinancialImpact(
            total_projected_loss_usd=float(total_projected_loss),
            potential_savings_usd=float(potential_savings),
            roi_improvement_pct=float(roi_improvement),
            avg_repair_cost_usd=self.avg_repair_cost
        )

    def generate_supplier_scorecard(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[SupplierMetrics]]:
        """
        Generate component-isolated supplier performance scorecard.

        CRITICAL: Calculates risk scores ONLY within each component category.
        Formula: (Claims for Supplier X / Total Units from Supplier X)

        Args:
            df: DataFrame with Warranty_Claim and supplier columns

        Returns:
            Dict mapping component category to List of SupplierMetrics
        """
        print("\n📋 Generating component-isolated supplier scorecard...")

        scorecard_by_category = {}

        # Process each component category separately
        for supplier_col, category_name in COMPONENT_CATEGORIES.items():
            if supplier_col not in df.columns:
                continue

            print(f"\n  Analyzing {category_name} suppliers...")

            # Calculate failure rate for each supplier in this category
            supplier_stats = df.groupby(supplier_col).agg({
                'Warranty_Claim': lambda x: (x == 'Yes').mean()  # Failure rate
            }).rename(columns={'Warranty_Claim': 'failure_rate'})

            supplier_counts = df[supplier_col].value_counts()

            # Calculate category average
            category_avg_failure_rate = (df['Warranty_Claim'] == 'Yes').mean()

            category_scorecard = []

            for supplier_name in supplier_stats.index:
                failure_rate = supplier_stats.loc[supplier_name, 'failure_rate']
                total_units = supplier_counts.get(supplier_name, 0)
                failures = int(total_units * failure_rate)
                financial_impact = failures * self.avg_repair_cost

                # Performance vs category average
                vs_category_avg = (failure_rate - category_avg_failure_rate) * 100

                # Risk categorization
                if failure_rate >= 0.25:
                    risk_level = "Critical"
                elif failure_rate >= 0.15:
                    risk_level = "High"
                elif failure_rate >= 0.10:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"

                # Flag critical cable suppliers (Cables-X equivalent)
                display_name = supplier_name
                if category_name == 'Cables' and risk_level in ["Critical", "High"]:
                    display_name = f"{supplier_name} ⚠️ CRITICAL"

                category_scorecard.append(SupplierMetrics(
                    supplier_name=display_name,
                    component_category=category_name,
                    failure_rate_pct=float(failure_rate * 100),
                    units_affected=failures,
                    total_units=int(total_units),
                    risk_level=risk_level,
                    financial_impact_usd=float(financial_impact),
                    vs_category_avg_pct=float(vs_category_avg)
                ))

            # Sort by failure rate within category
            category_scorecard.sort(key=lambda x: x.failure_rate_pct, reverse=True)
            scorecard_by_category[category_name] = category_scorecard

            # Print category summary
            critical_count = sum(1 for s in category_scorecard if s.risk_level == "Critical")
            high_count = sum(1 for s in category_scorecard if s.risk_level == "High")
            print(f"    ✓ {len(category_scorecard)} suppliers | "
                  f"Critical: {critical_count} | High: {high_count}")

        return scorecard_by_category

    def find_optimal_soldering_time(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Identify optimal Soldering_Time_s range where failures are minimized.

        Returns:
            (optimal_min, optimal_max, failure_rate_in_zone)
        """
        if 'Soldering_Time_s' not in df.columns:
            return (None, None, None)

        print("\n🔧 Analyzing optimal soldering time zone...")

        # Bin soldering times
        df['time_bin'] = pd.cut(df['Soldering_Time_s'], bins=20)

        # Calculate failure rate per bin
        bin_failures = df.groupby('time_bin')['Warranty_Claim'].apply(
            lambda x: (x == 'Yes').mean()
        )

        # Find bin with lowest failure rate (and sufficient samples)
        bin_counts = df['time_bin'].value_counts()
        valid_bins = bin_failures[bin_counts >= 20]  # At least 20 samples

        if len(valid_bins) == 0:
            return (None, None, None)

        optimal_bin = valid_bins.idxmin()
        optimal_failure_rate = valid_bins.min()

        optimal_min = optimal_bin.left
        optimal_max = optimal_bin.right

        print(f"✓ Optimal zone: {optimal_min:.1f}s - {optimal_max:.1f}s")
        print(f"  Failure rate in zone: {optimal_failure_rate*100:.2f}%")

        self.optimal_soldering_time = (optimal_min, optimal_max)

        return (optimal_min, optimal_max, optimal_failure_rate)

    def run_what_if_scenario(
        self,
        df: pd.DataFrame,
        optimize_soldering: bool = True,
        neutralize_cables_x: bool = True
    ) -> Dict:
        """
        What-If Engine: Calculate risk reduction from process optimization.

        ENHANCEMENT: Uses predicted probabilities instead of hard predictions
        for more precise risk measurement (Synergy Scenario).

        Scenarios:
        1. Optimize Soldering_Time_s to ideal range (exactly 3.5s for high-risk units)
        2. Replace Cables-X with best-performing supplier (WireTech)

        Args:
            df: Original data
            optimize_soldering: Apply soldering time optimization
            neutralize_cables_x: Set cable supplier to low-risk alternative

        Returns:
            Dict with scenario results and projected savings
        """
        print("\n🎯 Running What-If Scenario Analysis (Synergy Scenario)...")

        df_scenario = df.copy()
        y_true = df['Warranty_Claim']

        # Baseline: Use predicted probabilities for precise measurement
        X = df.drop('Warranty_Claim', axis=1)
        X = self.engineer_features(X)  # Add interaction features
        X_processed = self.preprocessor.transform(X)

        # Get probability of failure (class 1 = 'Yes')
        if hasattr(self.model, 'predict_proba'):
            baseline_probs = self.model.predict_proba(X_processed)[:, 1]
        else:
            # Fallback to binary predictions
            baseline_pred = self.model.predict(X_processed)
            baseline_probs = (baseline_pred == 'Yes').astype(float)

        baseline_risk_score = baseline_probs.sum()  # Sum of probabilities
        baseline_failures = int(baseline_probs.sum())  # Expected failures

        # Scenario 1: Optimize soldering time to exact target
        if optimize_soldering and self.optimal_soldering_time:
            optimal_min, optimal_max = self.optimal_soldering_time
            optimal_target = (optimal_min + optimal_max) / 2

            if 'Soldering_Time_s' in df_scenario.columns:
                # Target high-risk units (top 50% by current probability)
                high_risk_mask = baseline_probs >= np.median(baseline_probs)
                df_scenario.loc[high_risk_mask, 'Soldering_Time_s'] = optimal_target

                affected_units = high_risk_mask.sum()
                print(f"  Scenario: Soldering time optimized to {optimal_target:.2f}s for {affected_units:,} high-risk units")

        # Scenario 2: Replace Cables-X with best supplier
        if neutralize_cables_x:
            cable_col = 'Cable_Harness_Supplier'
            if cable_col in df_scenario.columns:
                # Find lowest-risk cable supplier
                supplier_risks = df.groupby(cable_col)['Warranty_Claim'].apply(
                    lambda x: (x == 'Yes').mean()
                )
                worst_supplier = supplier_risks.idxmax()
                best_supplier = supplier_risks.idxmin()

                df_scenario.loc[df_scenario[cable_col] == worst_supplier, cable_col] = best_supplier
                switched_units = (df[cable_col] == worst_supplier).sum()

                print(f"  Scenario: Replaced {worst_supplier} with {best_supplier} ({switched_units:,} units)")

        # Scenario predictions with probabilities
        X_scenario = df_scenario.drop('Warranty_Claim', axis=1)
        X_scenario = self.engineer_features(X_scenario)  # Add interaction features
        X_scenario_processed = self.preprocessor.transform(X_scenario)

        if hasattr(self.model, 'predict_proba'):
            scenario_probs = self.model.predict_proba(X_scenario_processed)[:, 1]
        else:
            scenario_pred = self.model.predict(X_scenario_processed)
            scenario_probs = (scenario_pred == 'Yes').astype(float)

        scenario_risk_score = scenario_probs.sum()
        scenario_failures = int(scenario_probs.sum())

        # Calculate impact using risk scores
        risk_score_reduction = baseline_risk_score - scenario_risk_score
        risk_reduction_pct = (risk_score_reduction / baseline_risk_score * 100) if baseline_risk_score > 0 else 0
        failures_prevented = baseline_failures - scenario_failures
        financial_savings = failures_prevented * self.avg_repair_cost

        results = {
            'baseline_failures': baseline_failures,
            'scenario_failures': scenario_failures,
            'failures_prevented': failures_prevented,
            'baseline_risk_score': float(baseline_risk_score),
            'scenario_risk_score': float(scenario_risk_score),
            'risk_reduction_pct': float(risk_reduction_pct),
            'financial_savings_usd': float(financial_savings),
            'scenario_applied': {
                'optimize_soldering': optimize_soldering,
                'neutralize_cables_x': neutralize_cables_x
            },
            'method': 'predicted_probabilities'  # Indicate we used probabilities
        }

        print(f"\n✅ What-If Results (Synergy Scenario):")
        print(f"  Baseline risk score: {baseline_risk_score:.1f} (expected failures: {baseline_failures:,})")
        print(f"  Optimized risk score: {scenario_risk_score:.1f} (expected failures: {scenario_failures:,})")
        print(f"  Risk reduction: {risk_reduction_pct:.1f}%")
        print(f"  Failures prevented: {failures_prevented:,}")
        print(f"  Projected savings: ${financial_savings:,.0f}")

        return results

    def generate_board_summary(
        self,
        risk_scores: List[RiskScore],
        supplier_scorecard: List[SupplierMetrics],
        financial_impact: FinancialImpact,
        what_if_results: Optional[Dict] = None
    ) -> str:
        """
        Generate Board-Ready executive summary with Top 3 Action Items.

        Returns:
            Formatted text summary for Board presentation
        """
        summary = []
        summary.append("=" * 80)
        summary.append("EXECUTIVE QUALITY DASHBOARD - BOARD SUMMARY")
        summary.append("USG Manufacturing Risk Assessment")
        summary.append("=" * 80)

        # Financial Overview
        summary.append("\n📊 FINANCIAL IMPACT:")
        summary.append(f"  • Total Projected Loss: ${financial_impact.total_projected_loss_usd:,.0f}")
        if financial_impact.potential_savings_usd > 0:
            summary.append(f"  • Potential Savings: ${financial_impact.potential_savings_usd:,.0f}")
            summary.append(f"  • ROI Improvement: {financial_impact.roi_improvement_pct:.1f}%")
        summary.append(f"  • Avg Repair Cost: ${financial_impact.avg_repair_cost_usd:,.0f}/unit")

        # Top Risk Drivers
        summary.append("\n🎯 TOP BUSINESS RISK DRIVERS:")
        for i, risk in enumerate(risk_scores[:3], 1):
            summary.append(f"\n  {i}. {risk.category} ({risk.contribution_pct:.1f}% of total risk)")
            summary.append(f"     Risk Score: {risk.risk_score:.1f}/100")
            summary.append(f"     Top Driver: {risk.top_drivers[0][0]}")
            summary.append(f"     → {risk.recommendation}")

        # Critical Suppliers
        summary.append("\n⚠️  CRITICAL SUPPLIERS:")
        critical_suppliers = [s for s in supplier_scorecard if s.risk_level == "Critical"][:3]
        for supplier in critical_suppliers:
            summary.append(f"  • {supplier.supplier_name}")
            summary.append(f"    Failure Rate: {supplier.failure_rate_pct:.1f}% | "
                          f"Financial Impact: ${supplier.financial_impact_usd:,.0f}")

        # What-If Scenario
        if what_if_results:
            summary.append("\n🎯 OPTIMIZATION POTENTIAL:")
            summary.append(f"  • Risk Reduction: {what_if_results['risk_reduction_pct']:.1f}%")
            summary.append(f"  • Projected Savings: ${what_if_results['financial_savings_usd']:,.0f}")

        # Top 3 Action Items
        summary.append("\n" + "=" * 80)
        summary.append("📋 TOP 3 ACTION ITEMS FOR THIS QUARTER:")
        summary.append("=" * 80)

        # Action 1: Highest risk category
        if len(risk_scores) > 0:
            top_risk = risk_scores[0]
            summary.append(f"\n1. ADDRESS {top_risk.category.upper()}")
            summary.append(f"   Priority: CRITICAL | Impact: {top_risk.contribution_pct:.1f}% risk reduction")
            summary.append(f"   Action: {top_risk.recommendation}")
            summary.append(f"   Timeline: 30 days | Owner: Operations VP")

        # Action 2: Critical suppliers
        if len(critical_suppliers) > 0:
            summary.append(f"\n2. SUPPLIER QUALITY INITIATIVE")
            summary.append(f"   Priority: HIGH | Suppliers affected: {len(critical_suppliers)}")
            summary.append(f"   Action: Audit {critical_suppliers[0].supplier_name} immediately")
            summary.append(f"   Expected savings: ${critical_suppliers[0].financial_impact_usd:,.0f}")
            summary.append(f"   Timeline: 45 days | Owner: Procurement VP")

        # Action 3: Process optimization
        if what_if_results and what_if_results['risk_reduction_pct'] > 10:
            summary.append(f"\n3. PROCESS OPTIMIZATION PROGRAM")
            summary.append(f"   Priority: MEDIUM | Risk reduction: {what_if_results['risk_reduction_pct']:.1f}%")
            summary.append(f"   Action: Implement optimized soldering parameters")
            summary.append(f"   Expected savings: ${what_if_results['financial_savings_usd']:,.0f}")
            summary.append(f"   Timeline: 60 days | Owner: Manufacturing Director")

        summary.append("\n" + "=" * 80)
        summary.append("End of Executive Summary")
        summary.append("=" * 80)

        return "\n".join(summary)

    def find_golden_zone_soldering(
        self,
        df: pd.DataFrame,
        target_reduction_pct: float = 87.0
    ) -> Dict:
        """
        Find the exact Soldering_Time_s range to hit the target risk reduction.
        This is the "Golden Zone" for production.

        Args:
            df: DataFrame with Soldering_Time_s and Warranty_Claim
            target_reduction_pct: Target risk reduction (default 87%)

        Returns:
            Dict with golden zone parameters
        """
        if 'Soldering_Time_s' not in df.columns:
            return None

        print(f"\n🎯 Finding Golden Zone for {target_reduction_pct}% risk reduction...")

        # Create fine-grained bins
        bins = np.linspace(df['Soldering_Time_s'].min(), df['Soldering_Time_s'].max(), 50)
        df_temp = df.copy()
        df_temp['time_bin'] = pd.cut(df_temp['Soldering_Time_s'], bins=bins)

        # Calculate failure rate per bin
        bin_stats = df_temp.groupby('time_bin').agg({
            'Warranty_Claim': [
                lambda x: (x == 'Yes').mean(),  # Failure rate
                'count'  # Sample size
            ]
        })

        bin_stats.columns = ['failure_rate', 'sample_size']
        bin_stats = bin_stats[bin_stats['sample_size'] >= 15]  # Min 15 samples

        if len(bin_stats) == 0:
            return None

        # Find zone with lowest failure rate
        best_bin = bin_stats['failure_rate'].idxmin()
        optimal_failure_rate = bin_stats.loc[best_bin, 'failure_rate']

        # Calculate baseline failure rate
        baseline_failure_rate = (df['Warranty_Claim'] == 'Yes').mean()

        # Calculate actual risk reduction
        actual_reduction = (baseline_failure_rate - optimal_failure_rate) / baseline_failure_rate * 100

        golden_zone = {
            'min_seconds': float(best_bin.left),
            'max_seconds': float(best_bin.right),
            'optimal_seconds': float((best_bin.left + best_bin.right) / 2),
            'failure_rate_pct': float(optimal_failure_rate * 100),
            'baseline_failure_rate_pct': float(baseline_failure_rate * 100),
            'risk_reduction_pct': float(actual_reduction),
            'sample_size': int(bin_stats.loc[best_bin, 'sample_size']),
            'meets_target': actual_reduction >= target_reduction_pct * 0.9  # Within 10%
        }

        print(f"✓ Golden Zone: {golden_zone['min_seconds']:.2f}s - {golden_zone['max_seconds']:.2f}s")
        print(f"  Optimal target: {golden_zone['optimal_seconds']:.2f}s")
        print(f"  Failure rate in zone: {golden_zone['failure_rate_pct']:.2f}%")
        print(f"  Risk reduction: {golden_zone['risk_reduction_pct']:.1f}%")

        if golden_zone['meets_target']:
            print(f"  ✅ MEETS TARGET ({target_reduction_pct}% goal)")
        else:
            print(f"  ⚠️  Below target (need {target_reduction_pct}%, achieved {actual_reduction:.1f}%)")

        return golden_zone

    def detect_interaction_effects(
        self,
        df: pd.DataFrame,
        high_risk_supplier: str = None
    ) -> Dict:
        """
        Detect if Cables-X failure is standalone or correlated with
        Soldering_Time_s anomalies (interaction effect).

        Args:
            df: DataFrame
            high_risk_supplier: Name of high-risk supplier (e.g., Cables-X)

        Returns:
            Dict with interaction analysis
        """
        print("\n🔬 Detecting interaction effects (Supplier × Soldering Time)...")

        cable_col = 'Cable_Harness_Supplier'
        solder_col = 'Soldering_Time_s'

        if cable_col not in df.columns or solder_col not in df.columns:
            return None

        # Identify high-risk cable supplier if not provided
        if high_risk_supplier is None:
            supplier_risks = df.groupby(cable_col)['Warranty_Claim'].apply(
                lambda x: (x == 'Yes').mean()
            )
            high_risk_supplier = supplier_risks.idxmax()

        print(f"  Analyzing: {high_risk_supplier}")

        # Create interaction groups
        df_temp = df.copy()

        # Bin soldering time into Low/Medium/High
        solder_terciles = pd.qcut(df_temp[solder_col], q=3, labels=['Low', 'Medium', 'High'])
        df_temp['solder_group'] = solder_terciles

        # Calculate failure rates for each combination
        interaction_matrix = df_temp.groupby([cable_col, 'solder_group'])['Warranty_Claim'].apply(
            lambda x: (x == 'Yes').mean()
        ).unstack(fill_value=0)

        # Focus on high-risk supplier
        if high_risk_supplier in interaction_matrix.index:
            high_risk_row = interaction_matrix.loc[high_risk_supplier]

            # Calculate interaction strength
            base_failure_rate = df_temp[df_temp[cable_col] == high_risk_supplier]['Warranty_Claim'].apply(
                lambda x: 1 if x == 'Yes' else 0
            ).mean()

            interaction_results = {
                'supplier': high_risk_supplier,
                'base_failure_rate_pct': float(base_failure_rate * 100),
                'low_solder_failure_pct': float(high_risk_row.get('Low', 0) * 100),
                'medium_solder_failure_pct': float(high_risk_row.get('Medium', 0) * 100),
                'high_solder_failure_pct': float(high_risk_row.get('High', 0) * 100),
                'interaction_detected': False,
                'interaction_strength': 0.0
            }

            # Check if interaction exists (variance across soldering groups)
            variance = high_risk_row.var()
            if variance > 0.01:  # Significant variance
                interaction_results['interaction_detected'] = True
                interaction_results['interaction_strength'] = float(variance)

                print(f"  ✅ INTERACTION DETECTED")
                print(f"     Low Solder Time: {interaction_results['low_solder_failure_pct']:.1f}% failure")
                print(f"     Medium Solder Time: {interaction_results['medium_solder_failure_pct']:.1f}% failure")
                print(f"     High Solder Time: {interaction_results['high_solder_failure_pct']:.1f}% failure")
                print(f"     → Soldering Time affects {high_risk_supplier} failure rate")
            else:
                print(f"  ℹ️  No significant interaction detected")
                print(f"     {high_risk_supplier} failure appears independent of soldering time")

            return interaction_results

        return None

    def generate_ceo_brief(
        self,
        df: pd.DataFrame,
        supplier_scorecard: Dict[str, List[SupplierMetrics]],
        golden_zone: Dict,
        what_if_results: Dict
    ) -> CEOBrief:
        """
        Generate CEO Executive Brief with key financial metrics.

        Returns:
            CEOBrief object with console-printable summary
        """
        print("\n" + "=" * 80)
        print("CEO EXECUTIVE BRIEF")
        print("=" * 80)

        # 1. Top technical cause
        if golden_zone and 'Soldering_Time_s' in df.columns:
            baseline_in_zone = df['Soldering_Time_s'].between(
                golden_zone['min_seconds'],
                golden_zone['max_seconds']
            ).mean() * 100
            top_technical = (f"Soldering Time out of Golden Zone "
                           f"({golden_zone['min_seconds']:.2f}s-{golden_zone['max_seconds']:.2f}s). "
                           f"Only {baseline_in_zone:.1f}% of production is currently in optimal range.")
        else:
            top_technical = "Assembly process variability"

        # 2. Top supplier financial leak
        all_suppliers = []
        for category, suppliers in supplier_scorecard.items():
            all_suppliers.extend(suppliers)

        all_suppliers.sort(key=lambda x: x.financial_impact_usd, reverse=True)

        if len(all_suppliers) > 0:
            worst_supplier = all_suppliers[0]
            top_supplier_leak = (f"{worst_supplier.supplier_name} ({worst_supplier.component_category}): "
                               f"{worst_supplier.failure_rate_pct:.1f}% failure rate = "
                               f"${worst_supplier.financial_impact_usd:,.0f}/year")
        else:
            top_supplier_leak = "Data unavailable"

        # 3. Supplier switch savings
        cables_scorecard = supplier_scorecard.get('Cables', [])
        if len(cables_scorecard) >= 2:
            worst_cable = cables_scorecard[0]
            best_cable = cables_scorecard[-1]

            # Calculate savings
            current_units = worst_cable.total_units
            current_failures = worst_cable.units_affected
            best_failure_rate = best_cable.failure_rate_pct / 100

            projected_failures = int(current_units * best_failure_rate)
            failures_saved = current_failures - projected_failures
            annual_savings = failures_saved * self.avg_repair_cost

            current_supplier_clean = worst_cable.supplier_name.replace(' ⚠️ CRITICAL', '')
            recommended_supplier_clean = best_cable.supplier_name

            # Net savings after one-time investment
            one_time_investment = 50000.0
            net_annual_savings = annual_savings - one_time_investment
            roi_first_year = (net_annual_savings / one_time_investment * 100) if one_time_investment > 0 else 0

        else:
            annual_savings = 0.0
            current_supplier_clean = "Unknown"
            recommended_supplier_clean = "Unknown"
            one_time_investment = 50000.0
            net_annual_savings = 0.0
            roi_first_year = 0.0

        # Risk reduction from what-if
        risk_reduction = what_if_results.get('risk_reduction_pct', 0.0) if what_if_results else 0.0

        brief = CEOBrief(
            top_technical_cause=top_technical,
            top_supplier_financial_leak=top_supplier_leak,
            supplier_switch_savings_usd=annual_savings,
            current_supplier=current_supplier_clean,
            recommended_supplier=recommended_supplier_clean,
            risk_reduction_pct=risk_reduction,
            one_time_investment_usd=one_time_investment,
            net_annual_savings_usd=net_annual_savings,
            roi_first_year_pct=roi_first_year
        )

        # Print CEO Brief to console
        print(f"\n1. TOP TECHNICAL CAUSE OF FAILURE:")
        print(f"   {brief.top_technical_cause}")

        print(f"\n2. TOP SUPPLIER-RELATED FINANCIAL LEAK:")
        print(f"   {brief.top_supplier_financial_leak}")

        print(f"\n3. SUPPLIER SWITCH FINANCIAL IMPACT:")
        print(f"   Current: {brief.current_supplier}")
        print(f"   Recommended: {brief.recommended_supplier}")
        print(f"   Annual Savings: ${brief.supplier_switch_savings_usd:,.0f}")
        print(f"   One-time Investment: ${brief.one_time_investment_usd:,.0f}")
        print(f"   Net First-Year Savings: ${brief.net_annual_savings_usd:,.0f}")
        print(f"   ROI (First Year): {brief.roi_first_year_pct:.1f}%")

        print(f"\n4. COMBINED OPTIMIZATION IMPACT:")
        print(f"   Risk Reduction: {brief.risk_reduction_pct:.1f}%")
        print(f"   → Supplier switch + Golden Zone compliance")

        print("\n" + "=" * 80)
        print("END OF CEO BRIEF")
        print("=" * 80 + "\n")

        return brief


# ============================================================================
# BALANCED RANDOM FOREST MODEL TRAINING
# ============================================================================

def train_balanced_rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 15,
    random_state: int = 42
) -> BalancedRandomForestClassifier:
    """
    Train Balanced Random Forest to handle class imbalance in warranty claims.

    Args:
        X_train: Training features
        y_train: Training labels (Yes/No)
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Trained BalancedRandomForestClassifier
    """
    print(f"\n🌲 Training Balanced Random Forest...")
    print(f"  Trees: {n_estimators} | Max depth: {max_depth}")

    # Encode target
    y_encoded = (y_train == 'Yes').astype(int)

    # Train balanced model
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        sampling_strategy='auto',  # Balance classes automatically
        replacement=True,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_encoded)

    print(f"✓ Model trained successfully")

    return model
