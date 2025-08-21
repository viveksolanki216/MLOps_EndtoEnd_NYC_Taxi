
# Machine Learning Workflow Orchestration

## Workflow Orchestration
Is the process of managing and automating the execution of series of tasks or steps that form a workflow, especially when
the tasks are interdependent and need to be executed in a specific order.

**Key components of workflow orchestration include**
- Task Scheduling: Determining when and how tasks are executed
- Dependency Management: Handling dependencies between tasks to ensure they are executed in the correct order
- Failure Recovery: Retry and alert
- Parallel Execution: when possible
- Monitoring and logging

**Popular Tools**
- Apache Airflow: Open-source platform, DAG based, Good for complex workflows
- Prefect 
- Dagster 
- Mage

## Machine Learning Pipelines
Sequence of automated ML tasks that are executed in a specific order to build, train, and deploy machine learning models.
ML workflows are often repetitve and can be automated to ensure consistency and efficiency. ML Pipelines form the 
backbone of reliable, reproducible, scalable machine learning workflows.

**What ML pipelines can do?**
- Automate the end-to-end ML workflow
- Scheduling
- Input/Output Tracking: Track data versions, code versions, and model versions, model parameters, and metrics
- Reproducibility: Ensure that the same pipeline can be run with the same results
- Lineage and Provenance: Track the lineage of data, models, and code, ensuring compliance and traceability
- Scalability: Parallielize tasks, run on distributed systems/cloud infrastructure
- Monitoring & Logging: Monitor pipeline execution, log metrics, and artifacts for debugging and analysis
- Collaboration: Share pipelines, components and models across teams and environments
- Modularity and Reuse: Defined standard steps  and reuse across steps

**Pipeline Steps can include:**
- Data Ingestion 
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Deployment
- Model Monitoring


## Key Differences between Workflow Orchestration (WO) and ML Pipelines (MLP)
- **Scope**: WO any general series of tasks, while MLP focus specifically on ML tasks.
- **Complexity**: WO handles complex dependencies and scheduling, while MLP are often linear or DAG-based.
- **Tools**: WO tools (e.g., Airflow, Prefect) can be used to manage ML pipelines, but ML-specific tools (e.g., Kubeflow, TFX) are designed for end-to-end ML workflows.


## Apache Airflow for ML Pipelines
Is a general-purpose complex workflow orchestration tool that can also be used to manage and automate ML pipelines. 

Youtube Tutorial Refernce: https://www.youtube.com/watch?v=mtJHMdoi_Gg&list=PLwFJcsJ61oujAqYpMp1kdUBcPG0sE0QMT&index=4

**Important concepts:**
- **DAG (Directed Acyclic Graph)**: Represents the workflow as a series of tasks and their dependencies
- **Tasks**: Individual steps in the workflow, such as data ingestion, preprocessing, training, etc
- **Operators**: BashOperator, PythonOperator, etc. Each task is defined using an operator that specifies what to do

**Important Notes:**
- Don't use Xcom for heavy data, it will take a lot of time to transfer data between tasks. Instead, use files or databases to store and share data between tasks.
- Return only serializable objects from tasks, such as strings, integers, lists, or dictionaries. Avoid returning complex objects like DataFrames or custom classes.
- All task can return or accept only one xcom object, so you need to use a dictionary or a list to return multiple values from a task.

### Pros for using Airflow for ML pipelines: (Mostly good for data engineering tasks)
- Automation and Sehduling
- Sequencing of complex workflows
- Handles retries, timeouts, SLAs
- Logging execution history
- Web UI for monitoring and debugging

### Cons for using Airflow for ML pipelines:
- No Code versioning: Does not take snapshot of code that was run to execute the task/pipeline
- Doesn't store input/output artifacts
- No Lineage tracking: Does not track the lineage of data, models, and code
- No experiment tracking or model management

### You need to use additional tools to fix the issues (cons mentioned above)
- MLflow for experiment tracking, model management, and lineage tracking
- DVC for data versioning and model versioning
- Git for code versioning
- Docker for reproducibility 
- Minio or S3 for artifact storage

### Or use a dedicated ML pipeline tool 
- Metaflow: 
  - developed by Netflix, open-source, easy to use, integrates with AWS
  - Pythonic API, supports data versioning, model management, and lineage tracking
- Kubeflow 



