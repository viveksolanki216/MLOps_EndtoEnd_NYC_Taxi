
# Airflow
Is a general-purpose complex workflow orchestration tool that can also be used to manage and automate ML pipelines. 

Youtube Tutorial Refernce: https://www.youtube.com/watch?v=mtJHMdoi_Gg&list=PLwFJcsJ61oujAqYpMp1kdUBcPG0sE0QMT&index=4

Important concepts:
- **DAG (Directed Acyclic Graph)**: Represents the workflow as a series of tasks and their dependencies
- **Tasks**: Individual steps in the workflow, such as data ingestion, preprocessing, training, etc
- **Operators**: BashOperator, PythonOperator, etc. Each task is defined using an operator that specifies what to do

Important Notes:
- Don't use Xcom for heavy data, it will take a lot of time to transfer data between tasks. Instead, use files or databases to store and share data between tasks.
- Return only serializable objects from tasks, such as strings, integers, lists, or dictionaries. Avoid returning complex objects like DataFrames or custom classes.
- All task can return or accept only one xcom object, so you need to use a dictionary or a list to return multiple values from a task.

# Pros for using Airflow for ML pipelines: (Mostly good for data engineering tasks)
- Automation and Sehduling
- Sequencing of complex workflows
- Handles retries, timeouts, SLAs
- Logging execution history
- Parallel execution of tasks
- Web UI for triggering dags, monitoring and debugging
- Rerun only failed tasks to minimize cost and time

# Cons for using Airflow for ML pipelines:
- No Code versioning: Does not take snapshot of code that was run to execute the task/pipeline
- Doesn't store input/output artifacts






