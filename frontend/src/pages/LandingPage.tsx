import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Upload, Play, TrendingUp, Zap, Shield, BarChart3 } from 'lucide-react';
import { pageVariants, staggerContainer, fadeIn } from '@/utils/animations';
import { useState, useEffect } from 'react';

export default function LandingPage() {
  const navigate = useNavigate();
  const [devicesCount, setDevicesCount] = useState(0);
  const [accuracy, setAccuracy] = useState(0);

  // Animated counters
  useEffect(() => {
    const targetDevices = 2310;
    const targetAccuracy = 95.2;
    const duration = 2000;
    const steps = 60;

    let currentDevices = 0;
    let currentAccuracy = 0;

    const interval = setInterval(() => {
      currentDevices = Math.min(currentDevices + targetDevices / steps, targetDevices);
      currentAccuracy = Math.min(currentAccuracy + targetAccuracy / steps, targetAccuracy);

      setDevicesCount(Math.round(currentDevices));
      setAccuracy(parseFloat(currentAccuracy.toFixed(1)));

      if (currentDevices >= targetDevices && currentAccuracy >= targetAccuracy) {
        clearInterval(interval);
      }
    }, duration / steps);

    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 text-white overflow-hidden"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      {/* Background animated grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(99,102,241,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.05)_1px,transparent_1px)] bg-[size:50px_50px] [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]" />

      {/* Hero Section */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-32">
        <motion.div
          className="text-center"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          {/* Animated badge */}
          <motion.div
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-full mb-8"
            variants={fadeIn}
          >
            <Zap className="w-4 h-4 text-primary animate-pulse" />
            <span className="text-sm text-primary-light">Production-Grade ML System</span>
          </motion.div>

          {/* Main heading with gradient */}
          <motion.h1
            className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6"
            variants={fadeIn}
          >
            <span className="bg-gradient-to-r from-primary via-primary-light to-primary bg-[length:200%_auto] animate-gradient bg-clip-text text-transparent">
              AI-Powered
            </span>
            <br />
            <span className="text-white">Quality Prediction</span>
          </motion.h1>

          {/* Subheading */}
          <motion.p
            className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto mb-12"
            variants={fadeIn}
          >
            Predict warranty failures before they happen. Save costs, improve quality,
            and ship with confidence.
          </motion.p>

          {/* Stats cards */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto mb-12"
            variants={staggerContainer}
          >
            <motion.div
              className="bg-dark-700/50 backdrop-blur-sm border border-primary/20 rounded-2xl p-6 hover:border-primary/40 transition-all"
              variants={fadeIn}
              whileHover={{ scale: 1.02, y: -4 }}
            >
              <div className="text-5xl font-bold text-primary mb-2">
                {devicesCount.toLocaleString()}
              </div>
              <div className="text-gray-400">Devices Analyzed</div>
            </motion.div>

            <motion.div
              className="bg-dark-700/50 backdrop-blur-sm border border-success/20 rounded-2xl p-6 hover:border-success/40 transition-all relative overflow-hidden"
              variants={fadeIn}
              whileHover={{ scale: 1.02, y: -4 }}
            >
              <div className="text-5xl font-bold text-success mb-2">
                {accuracy}%
              </div>
              <div className="text-gray-400">Prediction Accuracy</div>
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-success/0 via-success/10 to-success/0"
                animate={{
                  x: ['-100%', '100%'],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
            </motion.div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            className="flex flex-col sm:flex-row gap-4 justify-center"
            variants={fadeIn}
          >
            <motion.button
              className="group relative px-8 py-4 bg-gradient-primary text-white rounded-xl font-semibold text-lg overflow-hidden shadow-lg shadow-primary/50"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/upload')}
            >
              <span className="relative z-10 flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload Production Data
              </span>
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-primary-dark to-primary"
                initial={{ x: '-100%' }}
                whileHover={{ x: 0 }}
                transition={{ duration: 0.3 }}
              />
            </motion.button>

            <motion.button
              className="px-8 py-4 bg-dark-700/50 backdrop-blur-sm border-2 border-primary/30 text-white rounded-xl font-semibold text-lg hover:border-primary/60 transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/dashboard')}
            >
              <span className="flex items-center gap-2">
                <Play className="w-5 h-5" />
                View Live Demo
              </span>
            </motion.button>
          </motion.div>
        </motion.div>

        {/* Features grid */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-32"
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
        >
          <FeatureCard
            icon={<TrendingUp className="w-8 h-8" />}
            title="Real-time Predictions"
            description="<100ms inference latency with production-grade FastAPI backend"
            color="primary"
          />
          <FeatureCard
            icon={<Shield className="w-8 h-8" />}
            title="Full Interpretability"
            description="SHAP explanations for every prediction with visual waterfall plots"
            color="success"
          />
          <FeatureCard
            icon={<BarChart3 className="w-8 h-8" />}
            title="Business Impact"
            description="60-80% reduction in warranty costs ($155K+ annual savings)"
            color="warning"
          />
        </motion.div>

        {/* Animated production line SVG */}
        <motion.div
          className="mt-32 relative h-32 overflow-hidden"
          variants={fadeIn}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
        >
          <svg className="w-full h-full" viewBox="0 0 1200 120">
            {/* Production line */}
            <line
              x1="0"
              y1="60"
              x2="1200"
              y2="60"
              stroke="rgba(99, 102, 241, 0.3)"
              strokeWidth="2"
            />

            {/* Animated devices */}
            {[0, 1, 2, 3, 4].map((i) => (
              <motion.g
                key={i}
                initial={{ x: -100 }}
                animate={{
                  x: [1 - 100, 1200],
                }}
                transition={{
                  duration: 8,
                  repeat: Infinity,
                  delay: i * 1.6,
                  ease: 'linear',
                }}
              >
                <rect
                  x="0"
                  y="40"
                  width="60"
                  height="40"
                  rx="4"
                  fill="rgba(99, 102, 241, 0.8)"
                  stroke="rgba(99, 102, 241, 1)"
                  strokeWidth="2"
                />
                <circle
                  cx="30"
                  cy="60"
                  r="4"
                  fill={i % 3 === 0 ? '#10B981' : '#6366F1'}
                  className="animate-pulse"
                />
              </motion.g>
            ))}
          </svg>
        </motion.div>
      </div>
    </motion.div>
  );
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'primary' | 'success' | 'warning';
}

function FeatureCard({ icon, title, description, color }: FeatureCardProps) {
  const colorClasses = {
    primary: 'border-primary/20 hover:border-primary/60 text-primary',
    success: 'border-success/20 hover:border-success/60 text-success',
    warning: 'border-warning/20 hover:border-warning/60 text-warning',
  };

  return (
    <motion.div
      className={`bg-dark-700/30 backdrop-blur-sm border ${colorClasses[color]} rounded-2xl p-8 transition-all`}
      variants={fadeIn}
      whileHover={{ scale: 1.05, y: -8 }}
    >
      <div className={`mb-4 ${colorClasses[color]}`}>{icon}</div>
      <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </motion.div>
  );
}
