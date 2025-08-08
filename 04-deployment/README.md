# Model Deployment

This module contains the Flask web service for serving the NYC Taxi trip duration prediction model.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required model files: `XGBoost_Best_Model.pkl` and `data_preprocessor_obj.pkl`

### Setup and Installation

1. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
cd 04-deployment/web-service-flask
pip install -r requirements.txt
```

3. **Set environment variables (optional):**
```bash
export MODEL_ARTIFACT_DIR="./path/to/models/"
export FLASK_DEBUG="False"
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="9696"
```

4. **Run the service:**
```bash
python src/predict.py
```

### 🐳 Docker Deployment

1. **Build Docker image:**
```bash
docker build . -t trip_duration_predictor_service:v1
```

2. **Run container:**
```bash
docker run -p 9696:9696 -e FLASK_DEBUG=False trip_duration_predictor_service:v1
```

### 📋 API Usage

#### Health Check
```bash
curl http://localhost:9696/health
```

#### Make Prediction
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PULocationID": 1,
    "DOLocationID": 2,
    "trip_distance": 5.2
  }'
```

**Response:**
```json
{
  "duration": 18.5
}
```

### 🔧 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_ARTIFACT_DIR` | `./` | Directory containing model files |
| `FLASK_DEBUG` | `False` | Enable debug mode (never use True in production) |
| `FLASK_HOST` | `0.0.0.0` | Host to bind the service |
| `FLASK_PORT` | `9696` | Port to bind the service |

### 🏭 Production Deployment

**Using Gunicorn (Recommended):**
```bash
pip install gunicorn
gunicorn --bind=0.0.0.0:9696 --workers=4 src.predict:app
```

**Performance Tips:**
- Use multiple workers (`--workers=4`) for better concurrency
- Models are loaded once at startup for optimal performance
- Monitor `/health` endpoint for service status

## 📊 About Flask

Flask is a lightweight Python web framework perfect for ML model deployment:

### ✅ Advantages
- **Low Code**: Minimal code to create APIs
- **Flexibility**: Works with any ML/DL framework
- **Ecosystem**: Easy integration with monitoring, authentication
- **Prototyping**: Great for quick demos and prototypes

### 🎯 Use Cases
- Serving models as REST APIs
- Batch scoring triggers
- MLOps metrics collection
- Real-time predictions

### ⚠️ Production Considerations
- Always use a WSGI server (Gunicorn/uWSGI) in production
- Never set `debug=True` in production
- Implement proper logging and monitoring
- Use load balancers for high availability