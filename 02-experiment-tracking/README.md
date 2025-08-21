# Topics Covered In this Section
- Introduction to MLFlow
- Experiment Tracking Using MLFlow
   - How to start a mlflow tracking server providing metadata store and artifacts store
   - How to log parameters, metrics, artifacts and models for a single run and multiple runs for hyperparameter tuning
   - How to use autologging
- Model Management/Registry Using MLFlow

## MLFlow
Open-source platform for managing ML life-cycle including experimentation, reproducibility, model registry and deployment

### Key-Use Cases
- Model Experimentation Tracking
  -	Reproducibility: Stores parameters, hyper-parameters, model performance metrics
  -	Lineage: Paths for input data, output model packages
  -	Compare different runs in a experiment i.e. for hyper-parameter optimization
- Model Management
  - Storing and versioning models
  - Registering models that are approved for deployment

Why it’s critical?
Before MLflow (and similar tools like DVC, Kubeflow, SageMaker MLOps), teams struggled with reproducibility, 
collaboration, and automation in ML workflows. These tools brought standardization, traceability, and automation,
which are now essential for production-grade ML.
-	We run the same jupyter notebook changing settings for model training, either save previous runs stats in excel sheets or do not track at all. It becomes chaotic to manage all these experiments given we can run 100s of model in today’s processing power era. We may not reproduce the previous good models or loose lineage of it.
-	Centralized view of experiment history: difficult to collaborate before, no other teams currently/in future can see the efforts that already made.

### Model Registry
A central repository for managing ML models, versions, and metadata.
 - Central Model Store: Share models across teams and tools
 - Version Control: Trace model versions
 - Model Lineage: Track model lineage, including code, data, and parameters
 - Stage Management: Deploy only "production" models
 - Review and Approval workflows: Add comments, document changes
 - Reproducibility: Link every model to it's MLFlow run (code, params, data etc)

Where Model Registry is used?
- CI/CD for ML: Register ->  Test ->  Promote Automatically
- Model Governance: Track model lineage, approvals, and versions
- Collaboration: Share models with other teams, tools, and environments. View + Comments on models


### Tracking Server
What is a tracking server & Why need a tracking server?
- A central API endpoint for logging and retrieving ML Experiments.
- Useful when you run  across multiple machines or in a distributed environment or team members.
- It provides a web UI to visualize and compare experiments.
- Starts a server usign `mlflow server` command.

MLFlow still logs data if the tracking server is not running?
- Yes, it will log to a local file by default. i.e. `./mlruns/` directory.
- We can still run mlflow.log_params, mlflow.log_metrics, etc. without a tracking server.
- And then see everything via the UI by running `mlflow ui` command.
- Recommended only if local dev, solo experiments, is not recommended for production use

### Metadata Store (Backend Store)
Saves metadata about experiments, runs, parameters, metrics, artifacts, etc. Keeps structured information.

- **Local Development** Solo User Experiments: 
  - File Store (default, if no backend store is specified):
    - `./mlruns/` directory
    - No concurrent write access, suitable for local development.
  - SQLite (default): 
    - `sqlite:///mlflow.db` (a serverless light weight database)
    - No concurrent write access, suitable for local development.
- **Intermediate** Small-Medium Teams Collaboration:
  - PostgreSQL, MySQL, etc.:
    - Production grade relational databases.
    - Allows concurrent writes 
    - `mlflow server --backend-store-uri postgresql://user:password@host:port/dbname`
- **Advanced Entrprise Grade** 
  - Remote SQL databases + Authentication

### Artifact Store
Stores output files of experiments: trained models (.pkl, .onnx, etc.), logs, plots, feature importance files.
 - Local Folder: `--default-artifact-root ./artifacts`

## Tracking Server Configuration

mlflow server \
--backend-store-uri sqlite:///backend.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0
--port 5000

#### If you want to run server in background use nohup with a log
nohup mlflow server  \
--backend-stroe-uri sqlite:///mlflow_pseudo_source_model.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 \
--port 5000 > mlflow.log 2>&1 &


#### Setting up a Tracking Server on an EC2 instance with PostgreSQL on AWS RDS and S3 for Artifacts
Check Scenario-3 Jupuy Notebook for a complete setup example.
