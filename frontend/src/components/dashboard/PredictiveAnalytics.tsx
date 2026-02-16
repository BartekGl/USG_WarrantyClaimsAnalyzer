import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { BarChart3, Grid3x3 } from 'lucide-react';
import { fadeIn } from '@/utils/animations';
import RiskHeatmap from './RiskHeatmap';
import FeatureImportanceChart from './FeatureImportanceChart';
import PredictionStream from './PredictionStream';

export default function PredictiveAnalytics() {
  const [activeView, setActiveView] = useState<'heatmap' | 'features'>('heatmap');

  return (
    <div className="space-y-6">
      {/* View toggle */}
      <div className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-1 inline-flex gap-1">
        <button
          className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
            activeView === 'heatmap'
              ? 'bg-primary text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          onClick={() => setActiveView('heatmap')}
        >
          <Grid3x3 className="w-4 h-4" />
          Risk Heatmap
        </button>
        <button
          className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
            activeView === 'features'
              ? 'bg-primary text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          onClick={() => setActiveView('features')}
        >
          <BarChart3 className="w-4 h-4" />
          Feature Importance
        </button>
      </div>

      {/* Main visualization */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <AnimatePresence mode="wait">
          {activeView === 'heatmap' ? (
            <motion.div
              key="heatmap"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <RiskHeatmap />
            </motion.div>
          ) : (
            <motion.div
              key="features"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <FeatureImportanceChart />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Real-time prediction stream */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <h3 className="text-lg font-semibold mb-4">Live Prediction Stream</h3>
        <PredictionStream />
      </motion.div>
    </div>
  );
}
