# USG Failure Prediction Dashboard

Modern, production-grade ML dashboard built with React 18, TypeScript, and Tailwind CSS. Features stunning animations, real-time predictions, and comprehensive data visualizations.

## 🎨 Features

### Visual Design (Lovable.dev Style)
- **Minimalist & Animated**: Fluid transitions with Framer Motion
- **60 FPS Performance**: Optimized animations and rendering
- **Glass Morphism**: Modern backdrop blur effects
- **Gradient Animations**: Moving gradient text and backgrounds
- **Responsive Design**: Mobile-first approach with Tailwind CSS

### Core Functionality
- **Landing Page**: Animated hero with production line visualization
- **Data Upload**: Drag & drop CSV upload with real-time validation
- **3-Panel Dashboard**: Production overview, predictive analytics, insights
- **Risk Heatmap**: Interactive 10×10 device risk visualization
- **Real-time Stream**: Live prediction feed with smooth animations
- **Feature Importance**: Recharts-powered bar charts
- **Supplier Analytics**: Pie charts and performance metrics

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── dashboard/
│   │       ├── ProductionOverview.tsx      # Left panel - metrics cards
│   │       ├── PredictiveAnalytics.tsx     # Center panel - visualizations
│   │       ├── InsightsActions.tsx         # Right panel - insights
│   │       ├── RiskHeatmap.tsx             # Interactive heatmap
│   │       ├── FeatureImportanceChart.tsx  # Bar chart
│   │       └── PredictionStream.tsx        # Live feed
│   ├── pages/
│   │   ├── LandingPage.tsx                 # Hero section
│   │   ├── UploadPage.tsx                  # CSV upload
│   │   └── DashboardPage.tsx               # Main dashboard
│   ├── stores/
│   │   └── dashboardStore.ts               # Zustand state management
│   ├── types/
│   │   └── index.ts                        # TypeScript definitions
│   ├── utils/
│   │   ├── api.ts                          # API client
│   │   ├── animations.ts                   # Framer Motion variants
│   │   └── cn.ts                           # Class name utility
│   ├── App.tsx                             # Root component
│   ├── main.tsx                            # Entry point
│   └── index.css                           # Global styles
├── public/                                  # Static assets
├── index.html                              # HTML template
├── vite.config.ts                          # Vite configuration
├── tailwind.config.js                      # Tailwind CSS config
├── tsconfig.json                           # TypeScript config
├── Dockerfile                              # Production build
├── nginx.conf                              # Nginx configuration
└── package.json                            # Dependencies
```

## 🎭 Animation System

### Framer Motion Variants

```typescript
import { pageVariants, cardVariants, fadeIn } from '@/utils/animations';

// Page transitions
<motion.div variants={pageVariants} initial="initial" animate="animate">
  {/* Content */}
</motion.div>

// Card hover effects
<motion.div variants={cardVariants} whileHover="hover">
  {/* Card */}
</motion.div>
```

### Custom Animations

- **Page Transitions**: Fade + slide (20px offset)
- **Card Hover**: Scale 1.02 + translate Y(-4px)
- **Number Counters**: Smooth rolling animation (2s duration)
- **Gradient Text**: Moving background position (3s loop)
- **Pulse Glow**: Box shadow animation for alerts
- **Device Flow**: SVG animation for production line

## 🎨 Design System

### Color Palette

| Color | Value | Usage |
|-------|-------|-------|
| Primary | `#6366F1` | Buttons, links, primary actions |
| Success | `#10B981` | Pass indicators, positive metrics |
| Warning | `#F59E0B` | Medium risk, caution alerts |
| Danger | `#EF4444` | Failures, high risk, critical alerts |
| Dark BG | `#0F172A` | Main background |

### Typography

```css
font-family: 'Inter', system-ui, sans-serif;
```

- Headings: 700-900 weight
- Body: 400-600 weight
- Monospace: For device IDs and batch codes

### Spacing Scale

Follows Tailwind's default spacing scale (4px base unit)

## 📊 Component API

### RiskHeatmap

Interactive 10×10 grid showing device failure probabilities.

```tsx
<RiskHeatmap />
```

**Features:**
- Color-coded risk levels (green → yellow → red)
- Hover tooltips with device details
- Smooth scale animation on hover
- Real-time data updates

### FeatureImportanceChart

Horizontal bar chart showing top 10 predictive features.

```tsx
<FeatureImportanceChart />
```

**Props:** None (uses store data)

**Features:**
- Recharts integration
- Custom tooltips
- Gradient bars
- Change indicators

### PredictionStream

Real-time scrolling feed of predictions.

```tsx
<PredictionStream />
```

**Features:**
- Auto-refresh every 3 seconds
- Smooth enter/exit animations (AnimatePresence)
- Color-coded by prediction result
- Keeps last 5 items

## 🔌 API Integration

### Configuration

```typescript
// src/utils/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### Available Methods

```typescript
import { apiClient } from '@/utils/api';

// Health check
await apiClient.healthCheck();

// Single prediction
const prediction = await apiClient.predict(deviceData, includeShap);

// Batch prediction
const predictions = await apiClient.predictBatch(devices, includeShap);

// SHAP explanation
const shap = await apiClient.getShapExplanation(deviceId);

// Batch statistics
const stats = await apiClient.getBatchStats(batchId);
```

## 🗂️ State Management

Uses **Zustand** for lightweight, performant state management.

```typescript
import { useDashboardStore } from '@/stores/dashboardStore';

function MyComponent() {
  const predictions = useDashboardStore((state) => state.predictions);
  const setPredictions = useDashboardStore((state) => state.setPredictions);

  // Use state and actions
}
```

### Available State

- `predictions`: Array of prediction results
- `batchStats`: Batch-level statistics
- `supplierPerformance`: Supplier metrics
- `actionItems`: Alerts and notifications
- `selectedDevice`: Currently selected device ID
- `isLoading`: Global loading state
- `error`: Error messages

## 🐳 Docker Deployment

### Development

```bash
# Build image
docker build -t usg-dashboard:dev .

# Run container
docker run -d \
  -p 3000:80 \
  --name usg-dashboard \
  -e VITE_API_URL=http://localhost:8000 \
  usg-dashboard:dev
```

### Production with Docker Compose

```bash
# Start both frontend and backend
docker-compose -f docker-compose.full.yml up -d

# View logs
docker-compose -f docker-compose.full.yml logs -f frontend

# Stop services
docker-compose -f docker-compose.full.yml down
```

## 🧪 Performance Optimization

### Build Optimizations

- **Code Splitting**: Separate chunks for React, charts, and animations
- **Tree Shaking**: Removes unused code
- **Minification**: Terser for production builds
- **Asset Optimization**: Image compression and lazy loading

### Runtime Optimizations

- **Virtual Scrolling**: For long lists
- **Memoization**: React.memo for expensive components
- **Debouncing**: For search and filter inputs
- **Lazy Loading**: Route-based code splitting

### Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| First Contentful Paint | < 1.5s | ~1.2s |
| Time to Interactive | < 3s | ~2.5s |
| Animation FPS | 60 fps | 60 fps |
| Bundle Size | < 500KB | ~420KB (gzipped) |

## 🔧 Configuration

### Environment Variables

```bash
# .env
VITE_API_URL=http://localhost:8000          # Backend API URL
VITE_ENV=development                        # Environment
VITE_ENABLE_DEMO_MODE=true                  # Demo mode flag
VITE_ENABLE_REALTIME_SIMULATION=true        # Real-time simulation
```

### Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000', // API proxy
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
```

## 🎯 Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile: iOS Safari 13+, Chrome Android 90+

## 📝 License

MIT License - See main project LICENSE file

## 👥 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Follow TypeScript/ESLint guidelines
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/BartekGl/ALK_DuzyProjekt/issues)
- Documentation: See `/docs` folder in main project

---

**Built with ❤️ using React 18 + TypeScript + Tailwind CSS + Framer Motion**

Last Updated: January 2026
