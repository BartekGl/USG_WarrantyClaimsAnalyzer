import { create } from 'zustand';
import type {
  Prediction,
  BatchStats,
  SupplierPerformance,
  ActionItem,
  TimelineEvent,
  RiskDevice,
  DeviceData,
} from '@/types';

interface DashboardState {
  // Data
  predictions: Prediction[];
  batchStats: BatchStats[];
  supplierPerformance: SupplierPerformance[];
  actionItems: ActionItem[];
  timelineEvents: TimelineEvent[];
  riskDevices: RiskDevice[];
  uploadedData: DeviceData[];

  // UI State
  selectedDevice: string | null;
  selectedBatch: string | null;
  isLoading: boolean;
  error: string | null;
  viewMode: 'dashboard' | 'comparison' | 'detail';
  comparisonBatches: string[];

  // Real-time simulation
  isSimulationRunning: boolean;
  simulationSpeed: number;

  // Actions
  setPredictions: (predictions: Prediction[]) => void;
  addPrediction: (prediction: Prediction) => void;
  setBatchStats: (stats: BatchStats[]) => void;
  setSupplierPerformance: (performance: SupplierPerformance[]) => void;
  setActionItems: (items: ActionItem[]) => void;
  addActionItem: (item: ActionItem) => void;
  resolveActionItem: (id: string) => void;
  setTimelineEvents: (events: TimelineEvent[]) => void;
  setRiskDevices: (devices: RiskDevice[]) => void;
  setUploadedData: (data: DeviceData[]) => void;

  setSelectedDevice: (deviceId: string | null) => void;
  setSelectedBatch: (batchId: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setViewMode: (mode: 'dashboard' | 'comparison' | 'detail') => void;
  setComparisonBatches: (batches: string[]) => void;

  startSimulation: () => void;
  stopSimulation: () => void;
  setSimulationSpeed: (speed: number) => void;

  reset: () => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  // Initial state
  predictions: [],
  batchStats: [],
  supplierPerformance: [],
  actionItems: [],
  timelineEvents: [],
  riskDevices: [],
  uploadedData: [],

  selectedDevice: null,
  selectedBatch: null,
  isLoading: false,
  error: null,
  viewMode: 'dashboard',
  comparisonBatches: [],

  isSimulationRunning: false,
  simulationSpeed: 1,

  // Actions
  setPredictions: (predictions) => set({ predictions }),

  addPrediction: (prediction) =>
    set((state) => ({
      predictions: [prediction, ...state.predictions].slice(0, 100), // Keep last 100
    })),

  setBatchStats: (stats) => set({ batchStats: stats }),

  setSupplierPerformance: (performance) => set({ supplierPerformance: performance }),

  setActionItems: (items) => set({ actionItems: items }),

  addActionItem: (item) =>
    set((state) => ({
      actionItems: [item, ...state.actionItems],
    })),

  resolveActionItem: (id) =>
    set((state) => ({
      actionItems: state.actionItems.map((item) =>
        item.id === id ? { ...item, resolved: true } : item
      ),
    })),

  setTimelineEvents: (events) => set({ timelineEvents: events }),

  setRiskDevices: (devices) => set({ riskDevices: devices }),

  setUploadedData: (data) => set({ uploadedData: data }),

  setSelectedDevice: (deviceId) => set({ selectedDevice: deviceId }),

  setSelectedBatch: (batchId) => set({ selectedBatch: batchId }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  setViewMode: (mode) => set({ viewMode: mode }),

  setComparisonBatches: (batches) => set({ comparisonBatches: batches }),

  startSimulation: () => set({ isSimulationRunning: true }),

  stopSimulation: () => set({ isSimulationRunning: false }),

  setSimulationSpeed: (speed) => set({ simulationSpeed: speed }),

  reset: () =>
    set({
      predictions: [],
      batchStats: [],
      supplierPerformance: [],
      actionItems: [],
      timelineEvents: [],
      riskDevices: [],
      uploadedData: [],
      selectedDevice: null,
      selectedBatch: null,
      isLoading: false,
      error: null,
      viewMode: 'dashboard',
      comparisonBatches: [],
      isSimulationRunning: false,
      simulationSpeed: 1,
    }),
}));
