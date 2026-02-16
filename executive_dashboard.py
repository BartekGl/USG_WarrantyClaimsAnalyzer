"""
Executive Quality Dashboard - Board-Ready Edition
CEO-level insights with component-isolated supplier analysis and Golden Zone targeting.

Features:
- VIEW A: Executive KPI Card with ROI calculations
- VIEW B: Segmented Supplier Scorecard (Cables, Batteries, PCBs)
- VIEW C: Golden Zone for Production (Soldering_Time_s)
- CEO Brief: Console output with top 3 financial metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Import business analytics module
from business_analytics import (
    BusinessRiskEngine,
    BUSINESS_PILLARS,
    COMPONENT_CATEGORIES,
    SupplierMetrics,
    CEOBrief
)

# Import for pickle compatibility
try:
    from ml_core import LabelEncoderPipeline
except ImportError:
    pass  # Already handled in business_analytics

# Page configuration
st.set_page_config(
    page_title="Board-Ready Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white; }
    .big-metric {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        margin: 15px 0;
    }
    .savings-metric {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        margin: 15px 0;
    }
    .critical-box {
        background-color: #7F1D1D;
        border-left: 6px solid #EF4444;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .golden-zone {
        background-color: #064E3B;
        border: 3px solid #10B981;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    }
    h1 { color: #60A5FA; font-weight: 700; }
    h2 { color: #93C5FD; font-weight: 600; margin-top: 40px; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_business_engine():
    """Load business risk engine"""
    try:
        with st.spinner("🚀 Loading Board Intelligence Engine..."):
            engine = BusinessRiskEngine()
            engine.load_models()
            return engine
    except Exception as e:
        st.error(f"❌ Failed to load engine: {e}")
        return None


def create_visual_validation_chart(df: pd.DataFrame, golden_zone: dict) -> go.Figure:
    """
    VISUAL VALIDATION: Dual-axis chart proving high data density + low risk in Golden Zone.

    Axis 1 (Histogram): Distribution of Soldering_Time_s showing data volume
    Axis 2 (Line): Claim Probability curve showing risk
    Annotation: "Peak Performance Area: Low Risk & High Volume"
    """
    if 'Soldering_Time_s' not in df.columns or golden_zone is None:
        return None

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # AXIS 1: Histogram (Data Distribution)
    fig.add_trace(
        go.Histogram(
            x=df['Soldering_Time_s'],
            nbinsx=40,
            name='Data Volume',
            marker=dict(color='#60A5FA', opacity=0.6),
            hovertemplate='Time: %{x:.2f}s<br>Count: %{y}<extra></extra>'
        ),
        secondary_y=False
    )

    # AXIS 2: Claim Probability Curve (Risk)
    bins = np.linspace(df['Soldering_Time_s'].min(), df['Soldering_Time_s'].max(), 30)
    df_temp = df.copy()
    df_temp['time_bin'] = pd.cut(df_temp['Soldering_Time_s'], bins=bins)

    bin_stats = df_temp.groupby('time_bin').agg({
        'Warranty_Claim': lambda x: (x == 'Yes').mean() * 100
    }).reset_index()
    bin_stats['bin_center'] = bin_stats['time_bin'].apply(lambda x: x.mid)

    fig.add_trace(
        go.Scatter(
            x=bin_stats['bin_center'],
            y=bin_stats['Warranty_Claim'],
            mode='lines+markers',
            name='Claim Risk (%)',
            line=dict(color='#EF4444', width=4),
            marker=dict(size=10, color='#DC2626', symbol='diamond'),
            hovertemplate='Time: %{x:.2f}s<br>Risk: %{y:.2f}%<extra></extra>'
        ),
        secondary_y=True
    )

    # Highlight Golden Zone with semi-transparent green box
    fig.add_vrect(
        x0=golden_zone['min_seconds'],
        x1=golden_zone['max_seconds'],
        fillcolor='#10B981',
        opacity=0.2,
        line=dict(color='#10B981', width=3),
        annotation_text=f"✓ PEAK PERFORMANCE AREA<br>Low Risk & High Volume<br>{golden_zone['min_seconds']:.2f}s - {golden_zone['max_seconds']:.2f}s",
        annotation_position="top",
        annotation=dict(
            font_size=14,
            font_color='#10B981',
            font=dict(weight='bold'),
            bgcolor='rgba(16, 185, 129, 0.2)',
            bordercolor='#10B981',
            borderwidth=2
        )
    )

    # Configure axes
    fig.update_xaxes(
        title_text="Soldering Time (seconds)",
        gridcolor='#2D3748'
    )

    fig.update_yaxes(
        title_text="<b>Data Volume</b> (Number of Units)",
        secondary_y=False,
        gridcolor='#2D3748'
    )

    fig.update_yaxes(
        title_text="<b>Claim Risk</b> (%)",
        secondary_y=True,
        gridcolor='#374151'
    )

    fig.update_layout(
        title=dict(
            text="📊 Visual Validation: Distribution vs. Risk (Dual-Axis Analysis)",
            font=dict(size=22, color='white', family='Arial Black')
        ),
        template='plotly_dark',
        height=550,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        plot_bgcolor='#1A202C',
        paper_bgcolor='#0E1117'
    )

    return fig


def create_golden_zone_chart(df: pd.DataFrame, golden_zone: dict) -> go.Figure:
    """
    VIEW C: Golden Zone chart showing Soldering_Time vs Claim Probability
    Highlights the exact range factory must maintain for -87% risk reduction
    """
    if 'Soldering_Time_s' not in df.columns or golden_zone is None:
        return None

    # Create bins for soldering time
    bins = np.linspace(df['Soldering_Time_s'].min(), df['Soldering_Time_s'].max(), 30)
    df_temp = df.copy()
    df_temp['time_bin'] = pd.cut(df_temp['Soldering_Time_s'], bins=bins)

    # Calculate failure probability per bin
    bin_stats = df_temp.groupby('time_bin').agg({
        'Warranty_Claim': lambda x: (x == 'Yes').mean() * 100
    }).reset_index()

    bin_stats['bin_center'] = bin_stats['time_bin'].apply(lambda x: x.mid)

    fig = go.Figure()

    # Plot failure probability curve
    fig.add_trace(go.Scatter(
        x=bin_stats['bin_center'],
        y=bin_stats['Warranty_Claim'],
        mode='lines+markers',
        name='Failure Probability',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8, color='#DC2626'),
        hovertemplate='Time: %{x:.2f}s<br>Failure Rate: %{y:.2f}%<extra></extra>'
    ))

    # Highlight Golden Zone
    fig.add_vrect(
        x0=golden_zone['min_seconds'],
        x1=golden_zone['max_seconds'],
        fillcolor='#10B981',
        opacity=0.25,
        line_width=0,
        annotation_text=f"✓ GOLDEN ZONE<br>{golden_zone['min_seconds']:.2f}s - {golden_zone['max_seconds']:.2f}s<br>Failure: {golden_zone['failure_rate_pct']:.2f}%",
        annotation_position="top",
        annotation=dict(font_size=14, font_color='#10B981', font=dict(weight='bold'))
    )

    # Add target line at golden zone failure rate
    fig.add_hline(
        y=golden_zone['failure_rate_pct'],
        line_dash="dash",
        line_color='#10B981',
        annotation_text=f"Target: {golden_zone['failure_rate_pct']:.2f}%",
        annotation_position="right"
    )

    # Add baseline line
    fig.add_hline(
        y=golden_zone['baseline_failure_rate_pct'],
        line_dash="dot",
        line_color='#EF4444',
        annotation_text=f"Baseline: {golden_zone['baseline_failure_rate_pct']:.1f}%",
        annotation_position="right"
    )

    fig.update_layout(
        title=dict(
            text=f"🎯 Golden Zone: Factory Must Maintain {golden_zone['min_seconds']:.2f}s - {golden_zone['max_seconds']:.2f}s",
            font=dict(size=22, color='white', family='Arial Black')
        ),
        xaxis_title="Soldering Time (seconds)",
        yaxis_title="Claim Probability (%)",
        template='plotly_dark',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(gridcolor='#2D3748'),
        yaxis=dict(gridcolor='#2D3748'),
        plot_bgcolor='#1A202C',
        paper_bgcolor='#0E1117'
    )

    return fig


def create_component_supplier_tabs(scorecard_by_category: dict) -> None:
    """
    VIEW B: Segmented Supplier Scorecard
    Separate tabs for Cables, Batteries, PCBs with component-isolated analysis
    """
    st.header("🏭 VIEW B: Segmented Supplier Scorecard")
    st.markdown("*Component-isolated risk analysis. Each supplier ranked ONLY within its category.*")

    # Create tabs for each component category
    tab_names = list(scorecard_by_category.keys())
    tabs = st.tabs(tab_names)

    for tab, category_name in zip(tabs, tab_names):
        with tab:
            suppliers = scorecard_by_category[category_name]

            if len(suppliers) == 0:
                st.warning(f"No suppliers found for {category_name}")
                continue

            # Highlight critical suppliers
            critical_suppliers = [s for s in suppliers if s.risk_level == "Critical"]

            if critical_suppliers:
                st.markdown(f"""
                <div class="critical-box">
                <h3>🚨 CRITICAL QUALITY OUTLIER</h3>
                <p><b>{critical_suppliers[0].supplier_name}</b></p>
                <p>Failure Rate: <b>{critical_suppliers[0].failure_rate_pct:.1f}%</b></p>
                <p>Financial Impact: <b>${critical_suppliers[0].financial_impact_usd:,.0f}/year</b></p>
                <p>vs. Category Average: <b>{critical_suppliers[0].vs_category_avg_pct:+.1f} percentage points</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Create bar chart for this category
            supplier_names = [s.supplier_name for s in suppliers]
            failure_rates = [s.failure_rate_pct for s in suppliers]
            colors = []

            for s in suppliers:
                if s.risk_level == "Critical":
                    colors.append('#7F1D1D')
                elif s.risk_level == "High":
                    colors.append('#92400E')
                elif s.risk_level == "Medium":
                    colors.append('#854D0E')
                else:
                    colors.append('#064E3B')

            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=supplier_names,
                x=failure_rates,
                orientation='h',
                marker_color=colors,
                text=[f'{fr:.1f}%' for fr in failure_rates],
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='<b>%{y}</b><br>Failure Rate: %{x:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text=f"{category_name} Supplier Performance",
                    font=dict(size=18, color='white')
                ),
                xaxis_title="Failure Rate (%)",
                template='plotly_dark',
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='#2D3748'),
                yaxis=dict(autorange='reversed'),
                plot_bgcolor='#1A202C',
                paper_bgcolor='#0E1117'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.subheader(f"📊 {category_name} Supplier Metrics")

            supplier_df = pd.DataFrame([
                {
                    'Supplier': s.supplier_name,
                    'Failure Rate (%)': f"{s.failure_rate_pct:.2f}",
                    'Total Units': f"{s.total_units:,}",
                    'Failures': s.units_affected,
                    'Risk Level': s.risk_level,
                    'Financial Impact ($)': f"${s.financial_impact_usd:,.0f}",
                    'vs. Avg (pp)': f"{s.vs_category_avg_pct:+.1f}"
                }
                for s in suppliers
            ])

            st.dataframe(supplier_df, use_container_width=True, height=300)


def create_interaction_heatmap(interaction_results: dict) -> go.Figure:
    """Create heatmap showing Supplier × Soldering Time interaction"""
    if interaction_results is None or not interaction_results.get('interaction_detected'):
        return None

    supplier = interaction_results['supplier']
    data = [
        [interaction_results['low_solder_failure_pct']],
        [interaction_results['medium_solder_failure_pct']],
        [interaction_results['high_solder_failure_pct']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[supplier],
        y=['Low Solder Time', 'Medium Solder Time', 'High Solder Time'],
        colorscale='Reds',
        text=[[f"{val:.1f}%"] for row in data for val in row],
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Failure Rate (%)")
    ))

    fig.update_layout(
        title=dict(
            text=f"🔬 Interaction Effect: {supplier} × Soldering Time",
            font=dict(size=18, color='white')
        ),
        template='plotly_dark',
        height=350,
        xaxis=dict(side='top'),
        plot_bgcolor='#1A202C',
        paper_bgcolor='#0E1117'
    )

    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.title("📊 Board-Ready Quality Dashboard")
    st.markdown("### CEO-Level Manufacturing Intelligence")
    st.markdown("---")

    # Load engine
    engine = load_business_engine()
    if engine is None:
        st.stop()

    # Load data
    try:
        with st.spinner("📂 Loading USG manufacturing data..."):
            df = engine.load_and_prepare_data()
    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Data file not found**

        Expected location: `data/raw/USG_Data_cleared.csv`

        Error: {e}
        """)
        st.stop()

    # Sidebar controls
    st.sidebar.header("⚙️ Executive Controls")

    avg_repair_cost = st.sidebar.number_input(
        "Average Repair Cost (USD)",
        min_value=500,
        max_value=5000,
        value=1200,
        step=100
    )
    engine.avg_repair_cost = avg_repair_cost

    one_time_investment = st.sidebar.number_input(
        "Process Calibration Investment (USD)",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=5000,
        help="One-time cost for equipment calibration and process optimization"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dashboard Features:**
    - Component-isolated supplier analysis
    - Golden Zone targeting (-87% risk)
    - CEO financial brief
    - Interaction effect detection
    - Visual validation (Distribution vs. Risk)
    """)

    # ========================================================================
    # RUN ANALYSES
    # ========================================================================

    with st.spinner("🔍 Running Board-level analyses..."):
        # Generate component-isolated supplier scorecard
        supplier_scorecard = engine.generate_supplier_scorecard(df)

        # Find Golden Zone
        golden_zone = engine.find_golden_zone_soldering(df, target_reduction_pct=87.0)

        # Run What-If scenario (Synergy Scenario with probabilities)
        what_if_results = engine.run_what_if_scenario(
            df,
            optimize_soldering=True,
            neutralize_cables_x=True
        )

        # Detect interaction effects
        interaction_results = engine.detect_interaction_effects(df)

        # Generate CEO Brief (prints to console)
        ceo_brief = engine.generate_ceo_brief(
            df,
            supplier_scorecard,
            golden_zone,
            what_if_results
        )

    # ========================================================================
    # SIDEBAR: CEO EXECUTIVE SUMMARY
    # ========================================================================

    st.sidebar.markdown("---")
    st.sidebar.header("📋 CEO Summary")

    st.sidebar.markdown(f"""
    **1. Top Technical Cause:**
    {ceo_brief.top_technical_cause[:80]}...

    **2. Top Supplier Leak:**
    {ceo_brief.top_supplier_financial_leak[:80]}...

    **3. ROI from Supplier Switch:**
    - Annual Savings: ${ceo_brief.supplier_switch_savings_usd:,.0f}
    - Net (After $50k): ${ceo_brief.net_annual_savings_usd:,.0f}
    - ROI: **{ceo_brief.roi_first_year_pct:.0f}%**

    **4. Total Risk Reduction:**
    **{ceo_brief.risk_reduction_pct:.1f}%** (Combined optimization)
    """)

    st.sidebar.success("✅ Analysis Complete")

    # ========================================================================
    # VIEW A: EXECUTIVE KPI CARD
    # ========================================================================

    st.header("💰 VIEW A: Executive KPI Card")

    # Calculate total potential savings
    current_failures = (df['Warranty_Claim'] == 'Yes').sum()
    current_loss = current_failures * avg_repair_cost

    if what_if_results:
        projected_failures = what_if_results['scenario_failures']
        projected_loss = projected_failures * avg_repair_cost
        total_savings = current_loss - projected_loss
        net_savings = total_savings - one_time_investment
        roi_first_year = (net_savings / one_time_investment * 100) if one_time_investment > 0 else 0
    else:
        total_savings = 0
        net_savings = 0
        roi_first_year = 0

    # Display KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="big-metric">
        <h3 style="color: #FCA5A5; margin: 0;">Current Annual Loss</h3>
        <h1 style="color: white; margin: 10px 0;">${current_loss:,.0f}</h1>
        <p style="color: #D1D5DB; margin: 0;">{current_failures:,} failures × ${avg_repair_cost}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="savings-metric">
        <h3 style="color: #6EE7B7; margin: 0;">Net Annual Savings</h3>
        <h1 style="color: white; margin: 10px 0;">${net_savings:,.0f}</h1>
        <p style="color: #D1D5DB; margin: 0;">After ${one_time_investment:,.0f} investment</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="savings-metric">
        <h3 style="color: #6EE7B7; margin: 0;">First-Year ROI</h3>
        <h1 style="color: white; margin: 10px 0;">{roi_first_year:,.0f}%</h1>
        <p style="color: #D1D5DB; margin: 0;">Payback in {(one_time_investment / total_savings * 12):.1f} months</p>
        </div>
        """, unsafe_allow_html=True)

    # Financial breakdown
    st.subheader("💵 Financial Impact Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Gross Annual Savings", f"${total_savings:,.0f}", help="Total savings before investment")
        st.metric("One-Time Investment", f"${one_time_investment:,.0f}", help="Equipment calibration cost")

    with col2:
        if what_if_results:
            st.metric("Risk Reduction", f"{what_if_results['risk_reduction_pct']:.1f}%", help="Combined optimization impact")
            st.metric("Failures Prevented", f"{what_if_results['failures_prevented']:,}", help="Annual failure reduction")

    st.markdown("---")

    # ========================================================================
    # VIEW B: SEGMENTED SUPPLIER SCORECARD
    # ========================================================================

    create_component_supplier_tabs(supplier_scorecard)

    st.markdown("---")

    # ========================================================================
    # VIEW C: GOLDEN ZONE FOR PRODUCTION
    # ========================================================================

    st.header("🎯 VIEW C: Golden Zone for Production")

    if golden_zone:
        # Display golden zone summary
        st.markdown(f"""
        <div class="golden-zone">
        <h3 style="color: #10B981; margin-top: 0;">✓ GOLDEN ZONE IDENTIFIED</h3>
        <h2 style="color: white; margin: 15px 0;">{golden_zone['min_seconds']:.2f}s - {golden_zone['max_seconds']:.2f}s</h2>
        <p style="color: #D1D5DB; font-size: 18px; margin: 0;">
        Factory must maintain soldering time within this range to achieve <b>-{golden_zone['risk_reduction_pct']:.0f}% risk reduction</b>
        </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Optimal Target",
                f"{golden_zone['optimal_seconds']:.2f}s",
                help="Ideal soldering time setting"
            )

        with col2:
            st.metric(
                "Failure Rate in Zone",
                f"{golden_zone['failure_rate_pct']:.2f}%",
                delta=f"-{golden_zone['risk_reduction_pct']:.0f}%",
                delta_color="normal",
                help="Failure rate when operating in Golden Zone"
            )

        with col3:
            baseline_in_zone = df['Soldering_Time_s'].between(
                golden_zone['min_seconds'],
                golden_zone['max_seconds']
            ).mean() * 100
            st.metric(
                "Current Compliance",
                f"{baseline_in_zone:.1f}%",
                help="% of current production in Golden Zone"
            )

        with col4:
            meets_target = "✅ YES" if golden_zone['meets_target'] else "⚠️ NO"
            st.metric(
                "Meets -87% Target",
                meets_target,
                help="Does Golden Zone achieve target risk reduction?"
            )

        # Golden Zone chart
        fig_golden = create_golden_zone_chart(df, golden_zone)
        if fig_golden:
            st.plotly_chart(fig_golden, use_container_width=True)

    else:
        st.warning("⚠️ Soldering_Time_s data not available for Golden Zone analysis")

    st.markdown("---")

    # ========================================================================
    # INTERACTION EFFECTS
    # ========================================================================

    if interaction_results and interaction_results.get('interaction_detected'):
        st.header("🔬 Root Cause Analysis: Interaction Effects")

        st.info(f"""
        **Finding:** {interaction_results['supplier']} failure rate varies significantly with Soldering Time

        This indicates the supplier issue is NOT standalone - it's correlated with
        process anomalies. Fixing soldering time may partially mitigate supplier risk.
        """)

        fig_interaction = create_interaction_heatmap(interaction_results)
        if fig_interaction:
            st.plotly_chart(fig_interaction, use_container_width=True)

        st.markdown("---")

    # ========================================================================
    # CEO BRIEF DISPLAY
    # ========================================================================

    st.header("📋 CEO Executive Brief")

    st.markdown("""
    *The following summary has been printed to the console for executive review.*
    """)

    st.code(f"""
CEO EXECUTIVE BRIEF
{'=' * 80}

1. TOP TECHNICAL CAUSE OF FAILURE:
   {ceo_brief.top_technical_cause}

2. TOP SUPPLIER-RELATED FINANCIAL LEAK:
   {ceo_brief.top_supplier_financial_leak}

3. SUPPLIER SWITCH FINANCIAL IMPACT:
   Current: {ceo_brief.current_supplier}
   Recommended: {ceo_brief.recommended_supplier}
   Annual Savings: ${ceo_brief.supplier_switch_savings_usd:,.0f}
   One-time Investment: ${ceo_brief.one_time_investment_usd:,.0f}
   Net First-Year Savings: ${ceo_brief.net_annual_savings_usd:,.0f}
   ROI (First Year): {ceo_brief.roi_first_year_pct:.1f}%

4. COMBINED OPTIMIZATION IMPACT:
   Risk Reduction: {ceo_brief.risk_reduction_pct:.1f}%
   → Supplier switch + Golden Zone compliance

{'=' * 80}
END OF CEO BRIEF
{'=' * 80}
    """, language='text')

    # Download CEO Brief
    st.download_button(
        label="📥 Download CEO Brief (TXT)",
        data=f"""CEO EXECUTIVE BRIEF

1. TOP TECHNICAL CAUSE: {ceo_brief.top_technical_cause}
2. TOP SUPPLIER LEAK: {ceo_brief.top_supplier_financial_leak}
3. SUPPLIER SWITCH SAVINGS: ${ceo_brief.supplier_switch_savings_usd:,.0f}
   Current: {ceo_brief.current_supplier} → Recommended: {ceo_brief.recommended_supplier}
   Net Savings: ${ceo_brief.net_annual_savings_usd:,.0f} | ROI: {ceo_brief.roi_first_year_pct:.0f}%
4. TOTAL RISK REDUCTION: {ceo_brief.risk_reduction_pct:.1f}%
""",
        file_name="CEO_Brief_Quality_Dashboard.txt",
        mime="text/plain"
    )

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 20px;'>
    <p><b>Board-Ready Quality Dashboard</b> | Component-Isolated Supplier Analysis | Golden Zone Targeting</p>
    <p>🔒 Confidential - Executive Use Only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
