import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { listItemVariants } from '@/utils/animations';

interface StreamItem {
  id: string;
  deviceId: string;
  prediction: 'Yes' | 'No';
  probability: number;
  timestamp: Date;
}

export default function PredictionStream() {
  const [items, setItems] = useState<StreamItem[]>([]);

  // Simulate real-time predictions
  useEffect(() => {
    const interval = setInterval(() => {
      const newItem: StreamItem = {
        id: `${Date.now()}`,
        deviceId: `DEV_${Math.floor(Math.random() * 1000)}`,
        prediction: Math.random() > 0.9 ? 'Yes' : 'No',
        probability: Math.random(),
        timestamp: new Date(),
      };

      setItems((prev) => [newItem, ...prev].slice(0, 5));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
      <AnimatePresence initial={false}>
        {items.map((item) => (
          <motion.div
            key={item.id}
            className={`p-4 rounded-lg border ${
              item.prediction === 'Yes'
                ? 'bg-danger/5 border-danger/30'
                : 'bg-success/5 border-success/30'
            }`}
            variants={listItemVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            layout
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-3">
                {item.prediction === 'Yes' ? (
                  <AlertCircle className="w-5 h-5 text-danger mt-0.5" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-success mt-0.5" />
                )}
                <div>
                  <div className="font-semibold text-sm">{item.deviceId}</div>
                  <div className="text-xs text-gray-400">
                    {item.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div
                  className={`text-sm font-semibold ${
                    item.prediction === 'Yes' ? 'text-danger' : 'text-success'
                  }`}
                >
                  {item.prediction === 'Yes' ? 'Failure' : 'Pass'}
                </div>
                <div className="text-xs text-gray-400">
                  {(item.probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
