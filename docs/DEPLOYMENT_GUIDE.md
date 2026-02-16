# Deployment Guide: USG Failure Prediction System

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Docker Deployment](#docker-deployment)
6. [API Usage](#api-usage)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **OS:** Linux, macOS, or Windows
- **Python:** 3.10 or higher
- **RAM:** Minimum 4GB, recommended 8GB
- **CPU:** Any modern CPU (no GPU required)
- **Disk Space:** 2GB for environment and model artifacts

### Software Dependencies
- Python 3.10+
- pip or conda package manager
- Git (for cloning repository)

## Installation

### Option 1: Using pip

```bash
# Clone repository
git clone https://github.com/BartekGl/USG_WarrantyClaimsAnalyzer.git
cd USG_WarrantyClaimsAnalyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone repository
git clone https://github.com/BartekGl/USG_WarrantyClaimsAnalyzer.git
cd USG_WarrantyClaimsAnalyzer

# Create conda environment
conda env create -f environment.yml
conda activate usg-failure-prediction
```

## Configuration

### Data Setup

1. **Place your data file:**
   ```bash
   # Copy your CSV data file to the project
   cp /path/to/USG_Data_cleared.csv data/raw/
   ```

2. **Verify data format:**
   - CSV file with headers
   - Contains all required columns (44 features + target)
   - No duplicate headers

### Model Training

Run the Jupyter notebooks in sequence to train the model:

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Feature_Engineering.ipynb
# 3. notebooks/03_Model_Training.ipynb
# 4. notebooks/04_SHAP_Analysis.ipynb
```

Alternatively, train using Python scripts:

```bash
python scripts/train_model.py --data data/raw/USG_Data_cleared.csv --output models/
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=models/model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl
SHAP_EXPLAINER_PATH=models/shap_explainer.pkl

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log
```

## Running the Application

### Development Mode

```bash
# Start API server
cd src
python api.py

# Or use uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
# Using uvicorn with multiple workers
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Or using gunicorn
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "preprocessor_loaded": true,
#   "shap_explainer_loaded": true,
#   "timestamp": "2026-01-25T..."
# }
```

## Docker Deployment

### Build Docker Image

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
# Build image
docker build -t usg-failure-prediction:latest .

# Run container
docker run -d \
  --name usg-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  usg-failure-prediction:latest

# Check logs
docker logs -f usg-api
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with Docker Compose:

```bash
docker-compose up -d
```

## API Usage

### Interactive Documentation

Access API documentation at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "device_data": {
      "Batch_ID": "BATCH_001",
      "Assembly_Temp_C": 22.5,
      "Humidity_Percent": 45.0,
      "Solder_Temp_C": 350.0,
      "Solder_Time_s": 3.2,
      "Torque_Nm": 2.5,
      "Gap_mm": 0.15,
      "Region": "EU"
    },
    "include_shap": true,
    "threshold": 0.5
  }'
```

Response:
```json
{
  "prediction": "No",
  "probability": 0.23,
  "confidence": 0.77,
  "threshold": 0.5,
  "shap_values": {
    "Supplier_A_Failure_Rate": -0.15,
    "Batch_Failure_Rate": -0.12,
    "Solder_Temp_C": -0.08,
    ...
  },
  "timestamp": "2026-01-25T10:30:45"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "devices": [
      {"Batch_ID": "BATCH_001", "Assembly_Temp_C": 22.5, ...},
      {"Batch_ID": "BATCH_002", "Assembly_Temp_C": 23.1, ...}
    ],
    "include_shap": false,
    "threshold": 0.5
  }'
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "device_data": {
            "Batch_ID": "BATCH_001",
            "Assembly_Temp_C": 22.5,
            "Humidity_Percent": 45.0,
            # ... other features
        },
        "include_shap": True,
        "threshold": 0.5
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Top SHAP features: {list(result['shap_values'].keys())[:5]}")
```

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

### Logging

Monitor application logs:

```bash
# If using Docker
docker logs -f usg-api

# If running directly
tail -f logs/api.log
```

### Performance Metrics

Track key metrics:
- **Latency:** Target <100ms per prediction
- **Throughput:** Monitor requests/second
- **Error Rate:** Should be <1%
- **Model Drift:** Weekly comparison of prediction distributions

### Monitoring Dashboard

Integrate with monitoring tools:
- **Prometheus:** Metrics collection
- **Grafana:** Visualization
- **ELK Stack:** Log aggregation

Example Prometheus metrics:
```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
```

## Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Symptom:** API returns 503 or model_loaded=false

**Solutions:**
```bash
# Verify model files exist
ls -lh models/model.pkl
ls -lh models/preprocessor.pkl

# Check file permissions
chmod 644 models/*.pkl

# Verify model compatibility
python -c "import joblib; model = joblib.load('models/model.pkl'); print('OK')"
```

#### 2. Memory Issues

**Symptom:** Container crashes or OOM errors

**Solutions:**
```bash
# Increase Docker memory limit
docker run -m 4g ...

# Use smaller batch sizes
# Reduce number of workers
uvicorn src.api:app --workers 2
```

#### 3. Slow Predictions

**Symptom:** Latency >100ms

**Solutions:**
- Pre-load SHAP explainer
- Disable SHAP for batch predictions
- Use model caching
- Optimize feature engineering pipeline

#### 4. Missing Dependencies

**Symptom:** Import errors

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify installation
pip list | grep xgboost
pip list | grep shap
```

### Debug Mode

Run API in debug mode:

```bash
# Set log level to DEBUG
uvicorn src.api:app --log-level debug --reload

# Enable Python debugging
export PYTHONDEBUG=1
python src/api.py
```

### Contact Support

If issues persist:
1. Check GitHub Issues: https://github.com/BartekGl/USG_WarrantyClaimsAnalyzer/issues
2. Review logs in `logs/` directory
3. Contact development team with:
   - Error messages
   - System information
   - Steps to reproduce

## Scaling

### Horizontal Scaling

Deploy multiple API instances behind a load balancer:

```bash
# Using Docker Swarm
docker service create \
  --name usg-api \
  --replicas 3 \
  --publish 8000:8000 \
  usg-failure-prediction:latest

# Using Kubernetes (example)
kubectl apply -f k8s/deployment.yaml
```

### Vertical Scaling

Increase resources per instance:
- Add more CPU cores → increase workers
- Add more RAM → enable larger batch sizes
- Use faster storage → reduce model loading time

## Security

### Best Practices

1. **Authentication:** Implement API key or OAuth
2. **Rate Limiting:** Prevent API abuse
3. **Input Validation:** Sanitize all inputs
4. **HTTPS:** Use SSL/TLS in production
5. **Secrets Management:** Use environment variables, not hardcoded values

### Example: API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secure-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

## Backup and Recovery

### Model Versioning

```bash
# Backup current model
cp models/model.pkl models/model_v1.0.0.pkl

# Tag with metadata
echo "v1.0.0 - Trained on 2026-01-25" > models/VERSION.txt
```

### Disaster Recovery

1. **Regular Backups:** Daily backups of `models/` directory
2. **Version Control:** All code in Git
3. **Model Registry:** Store models in artifact repository
4. **Rollback Plan:** Keep last 3 model versions

---

**Last Updated:** January 2026
**Maintained By:** USG Analytics Team


