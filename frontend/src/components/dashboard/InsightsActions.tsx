import { motion } from 'framer-motion';
import { AlertTriangle, TrendingUp, Package } from 'lucide-react';
import { staggerContainer, fadeIn } from '@/utils/animations';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

export default function InsightsActions() {
  const supplierData = [
    { name: 'Supplier A', value: 45, color: '#6366F1' },
    { name: 'Supplier B', value: 30, color: '#10B981' },
    { name: 'Supplier C', value: 15, color: '#F59E0B' },
    { name: 'Supplier D', value: 10, color: '#EF4444' },
  ];

  const actionItems = [
    {
      type: 'critical' as const,
      title: 'High Failure Rate',
      description: 'BATCH_PCB-N-2024-B: 15% failure detected',
      priority: 'high' as const,
    },
    {
      type: 'warning' as const,
      title: 'Supplier Alert',
      description: 'Supplier C quality degradation',
      priority: 'medium' as const,
    },
    {
      type: 'info' as const,
      title: 'Process Optimization',
      description: 'Temperature variance detected',
      priority: 'low' as const,
    },
  ];

  return (
    <motion.div
      className="space-y-6"
      variants={staggerContainer}
      initial="initial"
      animate="animate"
    >
      {/* Supplier Performance */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <h3 className="text-lg font-semibold mb-4">Supplier Performance</h3>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={supplierData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
            >
              {supplierData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#1E293B',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="grid grid-cols-2 gap-2 mt-4">
          {supplierData.map((supplier) => (
            <div key={supplier.name} className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: supplier.color }} />
              <span className="text-gray-300">{supplier.name}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Action Items */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <h3 className="text-lg font-semibold mb-4">Action Items</h3>
        <div className="space-y-3">
          {actionItems.map((item, idx) => (
            <motion.div
              key={idx}
              className={`p-4 rounded-lg border ${
                item.type === 'critical'
                  ? 'bg-danger/5 border-danger/30'
                  : item.type === 'warning'
                  ? 'bg-warning/5 border-warning/30'
                  : 'bg-primary/5 border-primary/30'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <div className="flex items-start gap-3">
                <AlertTriangle
                  className={`w-5 h-5 mt-0.5 ${
                    item.type === 'critical'
                      ? 'text-danger'
                      : item.type === 'warning'
                      ? 'text-warning'
                      : 'text-primary'
                  }`}
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-sm">{item.title}</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        item.priority === 'high'
                          ? 'bg-danger/20 text-danger'
                          : item.priority === 'medium'
                          ? 'bg-warning/20 text-warning'
                          : 'bg-primary/20 text-primary'
                      }`}
                    >
                      {item.priority}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400">{item.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Quick Stats */}
      <motion.div
        className="bg-dark-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
        variants={fadeIn}
      >
        <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-dark-700/30 rounded-lg">
            <div className="flex items-center gap-2">
              <Package className="w-4 h-4 text-primary" />
              <span className="text-sm">Active Batches</span>
            </div>
            <span className="text-sm font-semibold">12</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-dark-700/30 rounded-lg">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-success" />
              <span className="text-sm">Avg Quality Score</span>
            </div>
            <span className="text-sm font-semibold text-success">94.2%</span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
