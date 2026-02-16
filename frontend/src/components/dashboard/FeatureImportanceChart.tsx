import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function FeatureImportanceChart() {
  const data = [
    { feature: 'Supplier_A_Failure_Rate', importance: 0.18, change: 0.02 },
    { feature: 'Batch_Failure_Rate', importance: 0.15, change: -0.01 },
    { feature: 'Solder_Temp_C', importance: 0.12, change: 0.03 },
    { feature: 'Assembly_Temp_x_Humidity', importance: 0.11, change: 0.01 },
    { feature: 'Anomaly_Score', importance: 0.09, change: -0.02 },
    { feature: 'Batch_Size', importance: 0.08, change: 0.0 },
    { feature: 'Torque_Nm', importance: 0.07, change: 0.01 },
    { feature: 'Gap_mm', importance: 0.06, change: -0.01 },
    { feature: 'Humidity_Percent', importance: 0.05, change: 0.0 },
    { feature: 'Solder_Time_s', importance: 0.04, change: 0.01 },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Top 10 Feature Importance</h3>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 150, bottom: 5 }}>
          <XAxis type="number" domain={[0, 0.2]} tick={{ fill: '#9CA3AF', fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
            width={140}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1E293B',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
          />
          <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill="#6366F1" />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Change indicators */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="flex items-center gap-2 p-3 bg-success/10 border border-success/30 rounded-lg">
          <div className="w-2 h-2 bg-success rounded-full" />
          <span className="text-success-light">3 features trending up</span>
        </div>
        <div className="flex items-center gap-2 p-3 bg-danger/10 border border-danger/30 rounded-lg">
          <div className="w-2 h-2 bg-danger rounded-full" />
          <span className="text-danger-light">2 features trending down</span>
        </div>
      </div>
    </div>
  );
}
