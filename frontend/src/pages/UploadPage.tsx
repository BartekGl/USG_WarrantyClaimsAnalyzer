import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import { Upload, FileText, CheckCircle, AlertCircle, ArrowLeft } from 'lucide-react';
import { useDashboardStore } from '@/stores/dashboardStore';
import { pageVariants, shakeAnimation } from '@/utils/animations';
import type { DeviceData } from '@/types';

export default function UploadPage() {
  const navigate = useNavigate();
  const setUploadedData = useDashboardStore((state) => state.setUploadedData);

  const [uploadState, setUploadState] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<DeviceData[]>([]);
  const [fileName, setFileName] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setFileName(file.name);
    setUploadState('uploading');
    setError(null);
    setProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + 10, 90));
    }, 100);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        clearInterval(progressInterval);
        setProgress(100);

        try {
          // Validate and transform data
          const data = results.data.map((row: any, idx) => ({
            deviceId: row.Device_UUID || `device_${idx}`,
            batchId: row.Batch_ID || '',
            assemblyTempC: parseFloat(row.Assembly_Temp_C || 0),
            humidityPercent: parseFloat(row.Humidity_Percent || 0),
            solderTempC: parseFloat(row.Solder_Temp_C || 0),
            solderTimeS: parseFloat(row.Solder_Time_s || 0),
            torqueNm: parseFloat(row.Torque_Nm || 0),
            gapMm: parseFloat(row.Gap_mm || 0),
            region: row.Region || 'Unknown',
            ...row,
          })) as DeviceData[];

          if (data.length === 0) {
            throw new Error('No valid data found in CSV');
          }

          setPreview(data.slice(0, 10));
          setUploadedData(data);
          setUploadState('success');

          // Auto-navigate after success
          setTimeout(() => {
            navigate('/dashboard');
          }, 2000);
        } catch (err: any) {
          setError(err.message || 'Failed to parse CSV file');
          setUploadState('error');
          setProgress(0);
        }
      },
      error: (err) => {
        clearInterval(progressInterval);
        setError(err.message || 'Failed to read CSV file');
        setUploadState('error');
        setProgress(0);
      },
    });
  }, [navigate, setUploadedData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  return (
    <motion.div
      className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 text-white p-8"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <div className="max-w-4xl mx-auto">
        {/* Back button */}
        <motion.button
          className="flex items-center gap-2 text-gray-400 hover:text-white mb-8 transition-colors"
          onClick={() => navigate('/')}
          whileHover={{ x: -4 }}
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Home
        </motion.button>

        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4">Upload Production Data</h1>
          <p className="text-xl text-gray-400">
            Upload your CSV file to start predicting device failures
          </p>
        </div>

        {/* Upload zone */}
        <motion.div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
            transition-all duration-300
            ${
              isDragActive
                ? 'border-primary bg-primary/10 scale-105'
                : uploadState === 'error'
                ? 'border-danger bg-danger/5'
                : 'border-gray-600 hover:border-primary/50 hover:bg-dark-700/30'
            }
          `}
          animate={uploadState === 'error' ? shakeAnimation : {}}
          whileHover={uploadState === 'idle' ? { scale: 1.02 } : {}}
        >
          <input {...getInputProps()} />

          {uploadState === 'idle' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              <Upload className="w-16 h-16 mx-auto text-primary" />
              <div>
                <p className="text-xl font-semibold mb-2">
                  {isDragActive ? 'Drop your file here' : 'Drag & drop your CSV file'}
                </p>
                <p className="text-gray-400">or click to browse</p>
              </div>
            </motion.div>
          )}

          {uploadState === 'uploading' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              <FileText className="w-16 h-16 mx-auto text-primary animate-pulse" />
              <div>
                <p className="text-xl font-semibold mb-4">{fileName}</p>
                <div className="w-full bg-dark-700 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-primary"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-sm text-gray-400 mt-2">{progress}% uploaded</p>
              </div>
            </motion.div>
          )}

          {uploadState === 'success' && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="space-y-4"
            >
              <CheckCircle className="w-16 h-16 mx-auto text-success" />
              <div>
                <p className="text-xl font-semibold text-success mb-2">Upload Successful!</p>
                <p className="text-gray-400">Redirecting to dashboard...</p>
              </div>
            </motion.div>
          )}

          {uploadState === 'error' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              <AlertCircle className="w-16 h-16 mx-auto text-danger" />
              <div>
                <p className="text-xl font-semibold text-danger mb-2">Upload Failed</p>
                <p className="text-gray-400">{error}</p>
                <button
                  onClick={() => setUploadState('idle')}
                  className="mt-4 px-6 py-2 bg-primary rounded-lg hover:bg-primary-dark transition-colors"
                >
                  Try Again
                </button>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Format guide */}
        <motion.div
          className="mt-8 bg-dark-700/30 border border-gray-700 rounded-xl p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="text-lg font-semibold mb-4">CSV Format Requirements</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            {[
              'Batch_ID',
              'Assembly_Temp_C',
              'Humidity_Percent',
              'Solder_Temp_C',
              'Solder_Time_s',
              'Torque_Nm',
              'Gap_mm',
              'Region',
            ].map((col) => (
              <div key={col} className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full" />
                <span className="text-gray-300 font-mono">{col}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Preview */}
        {preview.length > 0 && (
          <motion.div
            className="mt-8 bg-dark-700/30 border border-gray-700 rounded-xl p-6 overflow-x-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h3 className="text-lg font-semibold mb-4">Data Preview (First 10 rows)</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3">Batch ID</th>
                  <th className="text-left py-2 px-3">Temp (°C)</th>
                  <th className="text-left py-2 px-3">Humidity (%)</th>
                  <th className="text-left py-2 px-3">Region</th>
                </tr>
              </thead>
              <tbody>
                {preview.map((row, idx) => (
                  <tr key={idx} className="border-b border-gray-800/50">
                    <td className="py-2 px-3 text-gray-300">{row.batchId}</td>
                    <td className="py-2 px-3 text-gray-300">{row.assemblyTempC.toFixed(1)}</td>
                    <td className="py-2 px-3 text-gray-300">{row.humidityPercent.toFixed(1)}</td>
                    <td className="py-2 px-3 text-gray-300">{row.region}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
