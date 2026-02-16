import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Settings } from 'lucide-react';
import { pageVariants } from '@/utils/animations';
import ProductionOverview from '@/components/dashboard/ProductionOverview';
import PredictiveAnalytics from '@/components/dashboard/PredictiveAnalytics';
import InsightsActions from '@/components/dashboard/InsightsActions';

export default function DashboardPage() {
  const navigate = useNavigate();

  return (
    <motion.div
      className="min-h-screen bg-dark-900 text-white"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {/* Header */}
      <div className="bg-dark-800/50 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.button
                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                onClick={() => navigate('/')}
                whileHover={{ x: -4 }}
              >
                <ArrowLeft className="w-5 h-5" />
              </motion.button>
              <div>
                <h1 className="text-2xl font-bold">USG Failure Prediction</h1>
                <p className="text-sm text-gray-400">Real-time Production Monitoring</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-success/10 border border-success/30 rounded-lg">
                <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                <span className="text-sm text-success-light">System Online</span>
              </div>

              <motion.button
                className="p-2 hover:bg-dark-700 rounded-lg transition-colors"
                whileHover={{ rotate: 90 }}
                transition={{ duration: 0.3 }}
              >
                <Settings className="w-5 h-5 text-gray-400" />
              </motion.button>
            </div>
          </div>
        </div>
      </div>

      {/* Main 3-panel layout */}
      <div className="max-w-[1920px] mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-120px)]">
          {/* LEFT: Production Overview */}
          <div className="lg:col-span-3 space-y-6 overflow-y-auto custom-scrollbar">
            <ProductionOverview />
          </div>

          {/* CENTER: Predictive Analytics */}
          <div className="lg:col-span-6 overflow-y-auto custom-scrollbar">
            <PredictiveAnalytics />
          </div>

          {/* RIGHT: Insights & Actions */}
          <div className="lg:col-span-3 space-y-6 overflow-y-auto custom-scrollbar">
            <InsightsActions />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
