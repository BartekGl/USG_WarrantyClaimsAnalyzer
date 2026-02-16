// Core Types for USG Failure Prediction Dashboard

export interface DeviceData {
  deviceId?: string;
  batchId: string;
  assemblyTempC: number;
  humidityPercent: number;
  solderTempC: number;
  solderTimeS: number;
  torqueNm: number;
  gapMm: number;
  region: string;
  [key: string]: any;
}

export interface Prediction {
  deviceId: string;
  prediction: 'Yes' | 'No';
  probability: number;
  confidence: number;
  threshold: number;
  timestamp: string;
  shapValues?: Record<string, number>;
}

export interface BatchStats {
  batchId: string;
  totalDevices: number;
  predictedFailures: number;
  passRate: number;
  avgProbability: number;
  topFailureReasons: Array<{
    feature: string;
    impact: number;
  }>;
  timestamp: string;
}

export interface SupplierPerformance {
  supplier: string;
  totalDevices: number;
  failureRate: number;
  avgQualityScore: number;
  trend: 'up' | 'down' | 'stable';
}

export interface MetricsCard {
  title: string;
  value: number | string;
  change?: number;
  trend?: 'up' | 'down' | 'stable';
  icon?: string;
  color?: 'primary' | 'success' | 'warning' | 'danger';
}

export interface TimelineEvent {
  timestamp: string;
  batchId: string;
  deviceCount: number;
  failureRate: number;
  status: 'processing' | 'completed' | 'flagged';
}

export interface RiskDevice {
  deviceId: string;
  batchId: string;
  riskLevel: 'low' | 'medium' | 'high';
  probability: number;
  position: { x: number; y: number };
}

export interface ActionItem {
  id: string;
  type: 'warning' | 'critical' | 'info';
  title: string;
  description: string;
  batchId?: string;
  priority: 'low' | 'medium' | 'high';
  timestamp: string;
  resolved: boolean;
}

export interface ChartDataPoint {
  name: string;
  value: number;
  fill?: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  change?: number;
}

export interface SankeyNode {
  name: string;
}

export interface SankeyLink {
  source: number;
  target: number;
  value: number;
}

export interface ComparisonData {
  batchId: string;
  metrics: Record<string, number>;
  devices: Prediction[];
}

export interface UploadState {
  isUploading: boolean;
  progress: number;
  error: string | null;
  data: DeviceData[] | null;
}
