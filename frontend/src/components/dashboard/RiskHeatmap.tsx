import { motion } from 'framer-motion';
import { useState } from 'react';
import { useDashboardStore } from '@/stores/dashboardStore';

export default function RiskHeatmap() {
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);

  // Generate sample risk data (10x10 grid)
  const generateRiskData = () => {
    const data = [];
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const risk = Math.random();
        data.push({
          x,
          y,
          risk,
          deviceId: `DEV_${y * 10 + x}`,
          batchId: `BATCH_${Math.floor(y / 2)}`,
        });
      }
    }
    return data;
  };

  const [riskData] = useState(generateRiskData());

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return '#10B981'; // Green
    if (risk < 0.6) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
  };

  const getRiskLabel = (risk: number) => {
    if (risk < 0.3) return 'Low';
    if (risk < 0.6) return 'Medium';
    return 'High';
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Device Risk Distribution</h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-success" />
            <span className="text-gray-400">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-warning" />
            <span className="text-gray-400">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-danger" />
            <span className="text-gray-400">High</span>
          </div>
        </div>
      </div>

      {/* Heatmap grid */}
      <div className="relative">
        <div className="grid grid-cols-10 gap-2">
          {riskData.map((cell, idx) => (
            <motion.div
              key={idx}
              className="aspect-square rounded-lg cursor-pointer relative"
              style={{
                backgroundColor: getRiskColor(cell.risk),
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: idx * 0.01 }}
              whileHover={{ scale: 1.2, zIndex: 10 }}
              onMouseEnter={() => setHoveredCell(cell)}
              onMouseLeave={() => setHoveredCell(null)}
            />
          ))}
        </div>

        {/* Tooltip */}
        {hoveredCell && (
          <motion.div
            className="absolute z-20 bg-dark-700 border border-gray-600 rounded-lg p-4 shadow-xl pointer-events-none"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              left: `${(hoveredCell.x / 10) * 100}%`,
              top: `${(hoveredCell.y / 10) * 100}%`,
              transform: 'translate(-50%, -120%)',
            }}
          >
            <div className="text-sm space-y-1">
              <div className="font-semibold">{hoveredCell.deviceId}</div>
              <div className="text-gray-400">{hoveredCell.batchId}</div>
              <div className="flex items-center gap-2 mt-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getRiskColor(hoveredCell.risk) }}
                />
                <span>
                  {getRiskLabel(hoveredCell.risk)} Risk ({(hoveredCell.risk * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
