

# MLOps Training Using NYC Taxi Dataset

This tutorial is based on MLops Zoomcamp course "https://github.com/DataTalksClub/mlops-zoomcamp/tree/main".

Demonstrates **production-ready** end-to-end MLOps workflow using the NYC Taxi dataset, covering data preparation, model training, evaluation, and deployment.

## 🛠️ Technologies & Tools

- **MLFlow** for experiment tracking and model management
- **Airflow** for workflow orchestration and automation
- **Docker** for containerization and reproducible deployments
- **Flask** for serving ML models as REST APIs
- **XGBoost** for trip duration prediction
- **Scikit-learn** for data preprocessing

## 📁 Project Structure

```
├── 01-intro/                  # Basic ML pipeline without MLOps
├── 02-experiment-tracking/    # MLFlow integration and model registry
├── 03-orchestration/         # Airflow DAGs and workflow automation
├── 04-deployment/            # Production-ready Flask API service
└── README.md
```

## 🚀 Quick Start

### 1. **Basic ML Pipeline** (`01-intro/`)
Simple NYC taxi duration prediction without MLOps practices.

### 2. **Experiment Tracking** (`02-experiment-tracking/`)
MLFlow integration for tracking experiments, models, and metrics.

### 3. **Workflow Orchestration** (`03-orchestration/`)
Airflow DAGs for automated ML pipeline execution.

### 4. **Model Deployment** (`04-deployment/`)
Production-ready Flask service with:
- ✅ **Error handling and logging**
- ✅ **Health check endpoints**
- ✅ **Environment-based configuration**
- ✅ **Input validation**
- ✅ **Performance optimization**
- ✅ **Docker containerization**

## 🏭 Production Features

This implementation includes production-ready improvements:

- **Security**: No hardcoded secrets, configurable debug mode
- **Reliability**: Comprehensive error handling and logging
- **Performance**: Optimized model loading and caching
- **Monitoring**: Health checks and structured logging
- **Documentation**: Complete setup and usage guides
- **Testing**: Demo scripts and validation tools

## 📖 Getting Started

Each module contains detailed README files with setup instructions:

1. Start with `01-intro/` for basic understanding
2. Progress through `02-experiment-tracking/` for MLOps practices
3. Explore `03-orchestration/` for workflow automation
4. Deploy with `04-deployment/` for production usage

## 🎯 Learning Outcomes

By completing this tutorial, you'll understand:

- **MLOps pipeline design** and best practices
- **Experiment tracking** with MLFlow
- **Workflow orchestration** with Airflow
- **Model deployment** strategies and patterns
- **Production considerations** for ML systems
- **Docker containerization** for ML services