import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Package } from 'lucide-react';
import { staggerContainer, fadeIn } from '@/utils/animations';
import { useEffect, useState } from 'react';

export default function ProductionOverview() {
  const [metrics, setMetrics] = useState({
    totalDevices: 2310,
    predictedFailures: 220,
    passRate: 90.48,
  });

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {/* Metrics Cards */}
      <MetricCard
        title="Total Devices"
        value={metrics.totalDevices.toLocaleString()}
        icon={<Package className="w-6 h-6" />}
        color="primary"
        trend={{ value: 12, direction: 'up' }}
      />

      <MetricCard
        title="Predicted Failures"
        value={metrics.predictedFailures}
        icon={<AlertTriangle className="w-6 h-6" />}
        color="danger"
        trend={{ value: 5, direction: 'down' }}
        pulse
      />

      <MetricCard
        title="Pass Rate"
        value={`${metrics.passRate.toFixed(2)}%`}
        icon={<CheckCircle className="w-6 h-6" />}
        color="success"
        trend={{ value: 2.3, direction: 'up' }}
      />

      {/* Production Timeline */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <h3 className="text-lg font-semibold mb-4">Production Timeline</h3>
        <div className="space-y-3">
          {[
            { time: '14:30', batch: 'BATCH_045', count: 50, status: 'completed' },
            { time: '14:15', batch: 'BATCH_044', count: 48, status: 'processing' },
            { time: '14:00', batch: 'BATCH_043', count: 52, status: 'flagged' },
          ].map((event, idx) => (
            <motion.div
              key={idx}
              className="flex items-center gap-3 p-3 bg-dark-700/30 rounded-lg hover:bg-dark-700/50 transition-colors"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  event.status === 'completed'
                    ? 'bg-success'
                    : event.status === 'processing'
                    ? 'bg-primary animate-pulse'
                    : 'bg-danger'
                }`}
              />
              <div className="flex-1">
                <div className="text-sm font-medium">{event.batch}</div>
                <div className="text-xs text-gray-400">{event.count} devices • {event.time}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'primary' | 'success' | 'warning' | 'danger';
  trend?: {
    value: number;
    direction: 'up' | 'down';
  };
  pulse?: boolean;
}

function MetricCard({ title, value, icon, color, trend, pulse }: MetricCardProps) {
  const colorClasses = {
    primary: {
      bg: 'bg-primary/10',
      border: 'border-primary/30',
      text: 'text-primary',
      glow: pulse ? 'animate-pulse-glow' : '',
    },
    success: {
      bg: 'bg-success/10',
      border: 'border-success/30',
      text: 'text-success',
      glow: '',
    },
    warning: {
      bg: 'bg-warning/10',
      border: 'border-warning/30',
      text: 'text-warning',
      glow: '',
    },
    danger: {
      bg: 'bg-danger/10',
      border: 'border-danger/30',
      text: 'text-danger',
      glow: pulse ? 'shadow-lg shadow-danger/50 animate-pulse' : '',
    },
  };

  const colors = colorClasses[color];

  return (
    <motion.div
      className={`bg-dark-800/50 backdrop-blur-sm border ${colors.border} rounded-xl p-6 ${colors.glow}`}
      variants={fadeIn}
      whileHover={{ scale: 1.02, y: -4 }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`${colors.bg} ${colors.text} p-3 rounded-lg`}>{icon}</div>
        {trend && (
          <div
            className={`flex items-center gap-1 text-sm ${
              trend.direction === 'up' ? 'text-success' : 'text-danger'
            }`}
          >
            {trend.direction === 'up' ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{trend.value}%</span>
          </div>
        )}
      </div>
      <div className="text-3xl font-bold mb-1">{value}</div>
      <div className="text-sm text-gray-400">{title}</div>
    </motion.div>
  );
}
