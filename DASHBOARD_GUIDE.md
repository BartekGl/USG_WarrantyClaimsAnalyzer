**Mission Control Dashboard** - Complete User Guide

Production-grade analytics dashboard with Lovable.dev aesthetic for USG failure prediction.

---

## 🚀 Quick Start

### Prerequisites

1. **Trained model** - Run training first:
   ```powershell
   python ml_core.py
   ```

2. **Dashboard dependencies** - Install Streamlit and visualization libraries:
   ```powershell
   pip install -r dashboard_requirements.txt
   ```

### Launch Dashboard

**Windows:**
```powershell
# Activate environment
venv\Scripts\activate

# Launch dashboard
scripts\dashboard.bat

# OR directly
streamlit run app.py
```

**Linux/Mac:**
```bash
# Activate environment
source venv/bin/activate

# Launch dashboard
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## 📊 Dashboard Features

### 1. **Overview Tab** 🎯

**What you see:**
- System metrics (devices, failures, features, accuracy)
- F1 Score gauge (color-coded performance)
- Performance metrics table
- Violin plots comparing healthy vs failed devices

**Use cases:**
- Quick health check of the model
- Monitor overall system performance
- Identify feature distribution differences

**Interactivity:**
- Hover over plots for detailed values
- Gauge updates based on model performance

---

### 2. **SHAP Explainability Tab** 🔍

#### Global Importance Sub-tab 🌍

**What it shows:**
- Top 15 most important features across ALL devices
- Mean absolute SHAP values (higher = more impact)

**How to use:**
1. View bar chart to see which features drive predictions
2. Download full report as CSV
3. Use insights for quality improvements (e.g., "Supplier X has high impact")

**Business value:**
- Identify critical production parameters
- Focus quality control efforts
- Guide supplier negotiations

#### Local Explanation Sub-tab 🔬

**What it shows:**
- SHAP waterfall for a specific device
- Prediction result (PASS/FAIL + probability)
- Actual outcome comparison

**How to use:**
1. Use slider to select device (0 to 2,309)
2. View waterfall: Red bars = increases risk, Green bars = decreases risk
3. Compare predicted vs actual outcome

**Example interpretation:**
```
Device #42: 87.3% failure probability

Waterfall shows:
+ Supplier_A_Failure_Rate: +0.45 (RED) → High-risk supplier
+ Solder_Temp_C: +0.23 (RED) → Temperature too high
- Torque_Nm: -0.18 (GREEN) → Good torque value

Prediction: FAILURE ⚠️
Actual: Yes (correctly predicted)
```

---

### 3. **Clustering & PCA Tab** 🌐

**What it does:**
- Reduces 40 features to 2D/3D visualization
- Groups devices into clusters
- Colors by failure risk

**Settings:**
- **Dimensions:** 2D (simpler) or 3D (more detail)
- **Clusters:** 2-8 groups (default: 4)

**How to use:**
1. Select dimension (2D/3D)
2. Choose number of clusters
3. Click "Run Clustering"
4. Explore scatter plot:
   - **Color:** Risk score (red = high risk, green = low risk)
   - **Shape:** Circle = healthy, Diamond = failed

**Business insights:**
- Identify "hidden" groups of high-risk devices
- Spot anomalies (outliers)
- Understand feature relationships

**Example findings:**
```
Cluster 1 (Top-left): Low risk, tight grouping
  → Standard production, good quality

Cluster 3 (Bottom-right): High risk, scattered
  → Parameter variations, supplier issues
  → ACTION: Investigate Batch IDs in this cluster
```

---

### 4. **Model Duel Tab** ⚔️

**What it compares:**
- XGBoost (our main model) vs Random Forest
- Performance metrics side-by-side
- Feature importance differences

**Sections:**

#### Performance Comparison 📊
- Shows F1, Precision, Recall, Accuracy, ROC-AUC
- Color-coded table (green = better)
- Winner announcement with score

**Typical results:**
```
XGBoost:       F1 = 0.7945 🥇
Random Forest: F1 = 0.7623 🥈

XGBoost wins by 3.2%
```

#### Feature Importance Duel 🎯
- Grouped bar chart showing top 10 features
- Blue bars = XGBoost importance
- Green bars = Random Forest importance

**Insights:**
- If both models agree → Feature definitely important
- If they disagree → Feature has non-linear effects (XGBoost better at capturing)

#### Correlation Heatmap 🔥
- Top 15 features by variance
- Interactive matrix showing dependencies
- Red = positive correlation, Blue = negative

**Use case:**
- Identify multicollinearity (avoid redundant sensors)
- Find unexpected relationships
- Guide feature engineering

---

### 5. **What-If Simulator Tab** 🎮

**Most powerful feature** - Real-time prediction testing!

**How it works:**
1. Adjust sliders for production parameters
2. Risk gauge updates instantly
3. SHAP chart shows which features matter most

**Example scenario:**
```
Baseline: 35% failure risk (🟡 MEDIUM)

Actions:
1. Increase Humidity from 45% → 60%
   → Risk jumps to 67% (🔴 HIGH)
   → SHAP shows Humidity contributed +0.25

2. Decrease Solder_Temp from 350°C → 340°C
   → Risk drops to 28% (🟢 LOW)
   → SHAP shows Temp contributed -0.32

Conclusion: Control humidity AND temperature together
```

**Use cases:**
- **Process optimization:** Find parameter sweet spots
- **Failure investigation:** Reproduce failure conditions
- **Training:** Show operators impact of parameters
- **Quality planning:** Simulate "what if" scenarios

**Real-world example:**
```
Scenario: "What if we switch to Supplier B?"

Steps:
1. Load a device from Supplier A
2. Change Supplier_A_Failure_Rate → 0
3. Change Supplier_B_Failure_Rate → 0.15
4. Observe risk change: 12% → 24%

Decision: Supplier B has higher risk, negotiate quality improvement
```

---

## 🎨 Design System (Lovable.dev Style)

### Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| **Primary** | `#6366F1` | Buttons, headers, accents |
| **Success** | `#10B981` | Healthy devices, positive metrics |
| **Warning** | `#F59E0B` | Medium risk, cautions |
| **Danger** | `#EF4444` | Failures, high risk, alerts |
| **Background** | `#0F172A` | Dark theme base |

### Glassmorphism Effects

All cards use:
- **Background:** `rgba(255, 255, 255, 0.05)` (translucent white)
- **Backdrop blur:** 10px
- **Border:** `rgba(255, 255, 255, 0.1)`
- **Hover:** Lift 4px + glow effect

### Animations

- **Page load:** Slide in from top (headers) and left (sections)
- **Hover:** Cards lift 4px with shadow glow
- **Metrics:** Pulse animation for critical values
- **Tabs:** Smooth color transition on selection

---

## 🎯 Sidebar - System Pipeline

**Visual flowchart** showing data journey:

```
📥 Raw Data (2,310 devices)
    ↓
⚙️ Preprocessing (40 features)
    ↓
🤖 XGBoost (Ensemble)
    ↓
🔍 SHAP (Interpreter)
    ↓
✅ Decision (Pass/Fail)
```

**Quick Stats:**
- Total devices
- Failures (count + percentage)

**Use:** Quick reference for system architecture

---

## 📈 Performance & Optimization

### Expected Load Times

| Component | First Load | Subsequent |
|-----------|-----------|------------|
| Dashboard startup | 3-5 sec | 1-2 sec (cached) |
| SHAP global importance | 10-15 sec | 2-3 sec |
| SHAP local explanation | 2-3 sec | <1 sec |
| PCA clustering | 5-8 sec | 1-2 sec |
| Model duel (first time) | 30-40 sec | 2-3 sec |
| What-if simulation | <1 sec | <1 sec |

### Caching Strategy

Streamlit caches:
- ✅ Analytics engine (loaded once)
- ✅ SHAP explainer (reused)
- ✅ Random Forest (trained once)
- ✅ PCA results (until settings change)

**Tip:** First run is slower due to model loading and RF training. Subsequent runs are fast!

---

## 🔧 Troubleshooting

### Issue: Dashboard won't start

**Error:** `streamlit: command not found`

**Solution:**
```powershell
pip install -r dashboard_requirements.txt
```

---

### Issue: Model not found

**Error:** `FileNotFoundError: models/model.pkl`

**Solution:**
```powershell
# Train model first
python ml_core.py
```

---

### Issue: Slow SHAP calculations

**Symptom:** Global importance takes >30 seconds

**Solutions:**
1. Reduce sample size (already optimized to 500)
2. Use local explanations instead (faster)
3. Close other applications (free up RAM)

---

### Issue: Blank charts

**Symptom:** Plotly charts don't render

**Solutions:**
```powershell
# Reinstall plotly
pip install --upgrade plotly
```

---

### Issue: Memory error

**Symptom:** Dashboard crashes on large operations

**Solutions:**
1. Restart dashboard
2. Reduce PCA samples (modify analytics_engine.py)
3. Close other browser tabs

---

## 💡 Tips & Best Practices

### For Business Users

1. **Start with Overview** - Get familiar with metrics
2. **Use SHAP Local** - Investigate specific failures
3. **Try What-If** - Understand parameter impacts
4. **Export Reports** - Download SHAP importance CSV

### For Data Scientists

1. **Check Model Duel** - Validate XGBoost vs RF
2. **Explore PCA** - Understand data structure
3. **Analyze Correlations** - Identify feature relationships
4. **Use Simulator** - Test edge cases

### For Quality Engineers

1. **Monitor Violin Plots** - Spot distribution shifts
2. **Track Global Importance** - Focus on top drivers
3. **Simulate Process Changes** - Before implementing
4. **Compare Clusters** - Identify problematic batches

---

## 📊 Sample Use Cases

### Use Case 1: Root Cause Analysis

**Scenario:** Batch PCB-N-2024-B has 15% failure rate

**Steps:**
1. Go to **SHAP Local Explanation**
2. Find devices from this batch
3. Analyze SHAP waterfall for failed devices
4. Identify common top features (e.g., Supplier_A)
5. Cross-check in **Clustering** tab
6. Generate action plan

**Result:** Identified supplier issue, switched to alternative

---

### Use Case 2: Process Optimization

**Scenario:** Want to reduce humidity in assembly area

**Steps:**
1. Go to **What-If Simulator**
2. Load typical device parameters
3. Gradually decrease humidity slider
4. Observe risk change
5. Find optimal humidity range (e.g., 35-45%)
6. Validate with SHAP chart

**Result:** Humidity reduction from 50% → 40% reduces risk by 12%

---

### Use Case 3: New Supplier Evaluation

**Scenario:** Considering Supplier D for components

**Steps:**
1. Go to **SHAP Global Importance**
2. Check if supplier features are important
3. Go to **Model Duel** → Feature Importance
4. Confirm supplier impact in both models
5. Go to **What-If Simulator**
6. Simulate switching suppliers
7. Compare risk changes

**Result:** Supplier D increases risk by 8%, negotiate better terms

---

## 🎓 Training Materials

### For New Users (30-minute session)

1. **Demo Overview tab** (5 min)
   - Show metrics and gauge
   - Explain F1 score

2. **Demo SHAP Local** (10 min)
   - Pick 3 devices (1 pass, 1 fail, 1 borderline)
   - Explain waterfall interpretation

3. **Demo What-If** (10 min)
   - Show parameter adjustments
   - Demonstrate risk changes

4. **Q&A** (5 min)

### For Advanced Users (1-hour session)

1. All above, plus:
2. **PCA Clustering** (15 min) - Explain clusters
3. **Model Duel** (10 min) - Discuss differences
4. **Correlation Heatmap** (10 min) - Identify relationships

---

## 🔐 Security & Access

### Data Privacy

- Dashboard runs **locally** (not cloud)
- Data stays on your machine
- No external API calls (except Plotly rendering)

### Access Control

- No built-in authentication (local use)
- For production deployment:
  - Use Streamlit Cloud with authentication
  - Deploy behind VPN
  - Add password protection (streamlit-authenticator)

---

## 🚀 Deployment Options

### Option 1: Local Use (Current)

```powershell
streamlit run app.py
```

**Pros:** Simple, private, no setup
**Cons:** Only you can access

---

### Option 2: Streamlit Cloud (Free)

1. Push code to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy in 1 click
4. Share link with team

**Pros:** Free, easy sharing, auto-updates
**Cons:** Public (or limited authentication)

---

### Option 3: Docker Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt -r dashboard_requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Deploy:
```bash
docker build -t usg-dashboard .
docker run -p 8501:8501 usg-dashboard
```

---

## 📞 Support

### Common Questions

**Q: Can I customize colors?**
**A:** Yes! Edit CSS in `app.py` under `apply_custom_css()`

**Q: Can I add more tabs?**
**A:** Yes! Follow the tab structure in `app.py`

**Q: Can I export charts?**
**A:** Yes! Plotly charts have built-in download (camera icon)

**Q: Does it work offline?**
**A:** Yes! All processing is local

---

## 🎉 Advanced Features

### Custom Metrics

Add your own KPIs in Overview tab:

```python
# In app.py, Tab 1
with col5:
    st.metric(
        "Custom KPI",
        calculate_custom_kpi(),
        help="Your custom metric"
    )
```

### Additional Visualizations

Add new charts using Plotly:

```python
fig = px.scatter(data, x='feature1', y='feature2')
st.plotly_chart(fig)
```

### Export Functionality

Add download buttons:

```python
csv = dataframe.to_csv(index=False)
st.download_button(
    "Download Data",
    data=csv,
    file_name="export.csv"
)
```

---

## 📚 Additional Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python/
- **SHAP Docs:** https://shap.readthedocs.io
- **Lovable.dev:** https://lovable.dev (design inspiration)

---

**Last Updated:** January 2026
**Version:** 1.0
**Maintained By:** USG Analytics Team

🎯 **Ready to explore?** Run `scripts\dashboard.bat` and dive in!
