import axios from 'axios';
import type { DeviceData, Prediction, BatchStats } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiClient = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Single prediction
  async predict(deviceData: DeviceData, includeShap = true): Promise<Prediction> {
    const response = await api.post('/predict', {
      device_data: deviceData,
      include_shap: includeShap,
      threshold: 0.5,
    });

    return {
      deviceId: deviceData.deviceId || `device_${Date.now()}`,
      prediction: response.data.prediction,
      probability: response.data.probability,
      confidence: response.data.confidence,
      threshold: response.data.threshold,
      timestamp: response.data.timestamp,
      shapValues: response.data.shap_values,
    };
  },

  // Batch prediction
  async predictBatch(devices: DeviceData[], includeShap = false): Promise<Prediction[]> {
    const response = await api.post('/predict/batch', {
      devices: devices.map((d) => ({
        Batch_ID: d.batchId,
        Assembly_Temp_C: d.assemblyTempC,
        Humidity_Percent: d.humidityPercent,
        Solder_Temp_C: d.solderTempC,
        Solder_Time_s: d.solderTimeS,
        Torque_Nm: d.torqueNm,
        Gap_mm: d.gapMm,
        Region: d.region,
      })),
      include_shap: includeShap,
      threshold: 0.5,
    });

    return response.data.predictions.map((pred: any, idx: number) => ({
      deviceId: devices[idx].deviceId || `device_${idx}`,
      prediction: pred.prediction,
      probability: pred.probability,
      confidence: pred.confidence,
      threshold: 0.5,
      timestamp: response.data.timestamp,
      shapValues: pred.shap_values,
    }));
  },

  // SHAP explanation
  async getShapExplanation(deviceId: string) {
    const response = await api.get(`/api/shap-explain/${deviceId}`);
    return response.data;
  },

  // Batch statistics
  async getBatchStats(batchId: string): Promise<BatchStats> {
    const response = await api.get(`/api/batch-stats/${batchId}`);
    return response.data;
  },

  // Model info
  async getModelInfo() {
    const response = await api.get('/model/info');
    return response.data;
  },

  // Features
  async getFeatures() {
    const response = await api.get('/features');
    return response.data;
  },
};

export default api;
