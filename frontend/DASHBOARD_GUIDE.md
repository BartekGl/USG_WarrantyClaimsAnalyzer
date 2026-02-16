# USG Failure Prediction Dashboard - Quick Start Guide

## 🎨 What You Get

A **stunning, production-grade dashboard** inspired by Lovable.dev with:

- 🚀 **60 FPS animations** powered by Framer Motion
- 📊 **Real-time visualizations** with interactive charts
- 🎯 **3-panel layout** for comprehensive production monitoring
- 📱 **Responsive design** that works on all devices
- ⚡ **<100ms API latency** for instant predictions

## 🏃 Quick Start (Development)

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies (one time only)
npm install

# 3. Set up environment
cp .env.example .env

# 4. Start development server
npm run dev
```

Dashboard will be available at **http://localhost:3000**

> **Note:** Make sure the backend API is running at `http://localhost:8000`

## 🐳 Quick Start (Docker - Full Stack)

Deploy the **complete system** (frontend + backend + ML model) in one command:

```bash
# From project root directory
docker-compose -f docker-compose.full.yml up -d
```

Access:
- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## 📖 User Journey

### 1. Landing Page (/)

**What you see:**
- Animated hero section with gradient text
- Rolling number counter: "2,310 devices analyzed"
- Pulse indicator: "95.2% accuracy achieved"
- SVG production line animation showing devices moving

**Actions:**
- Click "Upload Production Data" → Go to upload page
- Click "View Live Demo" → Go to dashboard with sample data

### 2. Upload Page (/upload)

**What you see:**
- Large drag & drop zone with hover effects
- CSV format requirements guide
- Real-time upload progress bar

**Actions:**
1. Drag CSV file or click to browse
2. Watch animated parsing progress
3. Preview first 10 rows of data
4. Auto-redirect to dashboard when complete

**Expected CSV format:**
```csv
Batch_ID,Assembly_Temp_C,Humidity_Percent,Solder_Temp_C,Solder_Time_s,Torque_Nm,Gap_mm,Region
BATCH_001,22.5,45.0,350.0,3.2,2.5,0.15,EU
BATCH_002,23.1,48.2,352.5,3.1,2.6,0.14,APAC
...
```

### 3. Dashboard (/dashboard)

**Layout:** 3-panel design

#### LEFT PANEL: Production Overview
- **Metrics Cards:**
  - Total Devices (with trend indicator)
  - Predicted Failures (pulsing red if high)
  - Pass Rate (percentage with trend)

- **Production Timeline:**
  - Recent batches with status indicators
  - Color-coded: Green (completed), Blue (processing), Red (flagged)
  - Click batch for details

#### CENTER PANEL: Predictive Analytics
- **View Toggle:**
  - Risk Heatmap (default)
  - Feature Importance Chart

- **Risk Heatmap (WOW Factor!):**
  - 10×10 grid of devices
  - Color: Green (low risk) → Yellow (medium) → Red (high)
  - Hover any cell for tooltip with:
    - Device ID
    - Batch ID
    - Risk level and percentage
  - Smooth scale animation on hover

- **Feature Importance Chart:**
  - Top 10 predictive features
  - Horizontal bar chart with Recharts
  - Tooltips showing exact values
  - Trend indicators (up/down/stable)

- **Live Prediction Stream:**
  - Scrolling feed of latest predictions
  - Updates every 3 seconds (simulated)
  - Color-coded by result (green/red)
  - Smooth enter/exit animations
  - Shows: Device ID, timestamp, prediction, probability

#### RIGHT PANEL: Insights & Actions
- **Supplier Performance:**
  - Animated donut chart
  - Breakdown by supplier
  - Color-coded segments

- **Action Items:**
  - Auto-generated alerts
  - Priority badges (high/medium/low)
  - Examples:
    - "⚠️ BATCH_PCB-N-2024-B: 15% failure rate"
    - "Supplier C quality degradation"
    - "Temperature variance detected"

- **Quick Stats:**
  - Active batches count
  - Average quality score
  - Updated in real-time

## 🎬 Animations in Action

### Page Transitions
All page changes feature:
- Fade in/out (0.5s duration)
- Slide up 20px on entry
- Smooth easing

### Card Interactions
Hover over any card to see:
- Scale to 1.02
- Translate Y -4px (lift effect)
- Smooth 0.3s transition

### Risk Heatmap
- Individual cells: Scale to 1.2 on hover
- Tooltip: Fade in with scale animation
- Grid: Stagger animation on load (0.01s per cell)

### Metric Cards
- Numbers: Rolling counter animation (2s)
- Pulse glow: Continuous for high-priority alerts
- Trend arrows: Color-coded (green ↑ / red ↓)

### Prediction Stream
- New items: Slide in from left
- Old items: Slide out to right
- Smooth layout animations with AnimatePresence

## 🎨 Design System

### Colors
```css
Primary:  #6366F1 (Indigo)  - Buttons, links
Success:  #10B981 (Green)   - Pass indicators
Warning:  #F59E0B (Amber)   - Medium risk
Danger:   #EF4444 (Red)     - Failures
Dark BG:  #0F172A (Slate)   - Background
```

### Typography
- Font: Inter (Google Fonts)
- Headings: 700-900 weight
- Body: 400-600 weight

### Animations
- Page transitions: 0.5s
- Hover effects: 0.2-0.3s
- Number counters: 2s
- Gradient text: 3s loop

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Backend API URL
VITE_API_URL=http://localhost:8000

# Environment
VITE_ENV=development

# Feature flags
VITE_ENABLE_DEMO_MODE=true
VITE_ENABLE_REALTIME_SIMULATION=true
```

## 📱 Responsive Breakpoints

- **Mobile:** < 640px
- **Tablet:** 640px - 1024px
- **Desktop:** > 1024px
- **Large Desktop:** > 1920px

Dashboard adapts:
- 3-panel → 1-panel on mobile
- Heatmap grid: 10×10 → 5×5 on mobile
- Charts: Responsive width/height

## 🚀 Performance Tips

### Development Mode
```bash
npm run dev
```
- Hot module replacement (HMR)
- Fast refresh
- Source maps enabled

### Production Build
```bash
npm run build
npm run preview  # Test production build locally
```

**Build optimizations:**
- Code splitting (React, charts, animations)
- Tree shaking (removes unused code)
- Minification with Terser
- Asset compression
- Bundle size: ~420KB gzipped

## 🐛 Troubleshooting

### Issue: API Connection Failed
**Solution:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify VITE_API_URL in `.env`
3. Check CORS settings in backend

### Issue: Blank Page After Build
**Solution:**
1. Check browser console for errors
2. Verify all assets in `dist/` folder
3. Test with `npm run preview` first

### Issue: Animations Stuttering
**Solution:**
1. Check CPU usage (target 60 FPS)
2. Reduce animation complexity in Chrome DevTools
3. Disable animations: Set `prefers-reduced-motion`

### Issue: Upload Not Working
**Solution:**
1. Verify CSV format matches requirements
2. Check file size (< 10MB recommended)
3. Inspect browser console for parsing errors

## 📊 Sample Data

For testing, create a CSV with this structure:

```csv
Batch_ID,Assembly_Temp_C,Humidity_Percent,Solder_Temp_C,Solder_Time_s,Torque_Nm,Gap_mm,Region
BATCH_001,22.5,45.0,350.0,3.2,2.5,0.15,EU
BATCH_002,23.1,48.2,352.5,3.1,2.6,0.14,APAC
BATCH_003,21.8,42.5,348.0,3.3,2.4,0.16,EU
```

## 🎯 Next Steps

1. **Explore the Dashboard:**
   - Upload sample CSV data
   - Interact with risk heatmap
   - Monitor live prediction stream

2. **Customize:**
   - Modify colors in `tailwind.config.js`
   - Adjust animations in `src/utils/animations.ts`
   - Add new features in `src/components/`

3. **Deploy to Production:**
   - Build with `npm run build`
   - Use Docker: `docker build -t usg-dashboard .`
   - Or deploy with `docker-compose.full.yml`

## 📞 Support

Need help? Check:
- **Frontend README:** `frontend/README.md`
- **Main Project Docs:** `/docs` folder
- **API Docs:** http://localhost:8000/docs (when backend running)

---

**Built with ❤️ for stunning ML visualizations**

Enjoy the dashboard! 🎉
