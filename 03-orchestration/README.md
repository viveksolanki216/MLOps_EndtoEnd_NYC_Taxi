
# Topics Covered In this Section
- Introduction to Airflow
- Running Apache Airflow Server on Docker
- Create ML Pipeline DAGs in Airflow

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

### Pros for using Airflow for ML pipelines: (Mostly good for data engineering tasks)
- Automation and Sehduling
- Sequencing of complex workflows
- Handles retries, timeouts, SLAs
- Logging execution history
- Parallel execution of tasks
- Web UI for triggering dags, monitoring and debugging
- Rerun only failed tasks to minimize cost and time

### Cons for using Airflow for ML pipelines:
- No Code versioning: Does not take snapshot of code that was run to execute the task/pipeline
- Doesn't store input/output artifacts


## Running Airflow on Docker
This install Airflow server on docker and docker-compose. That communicats via a REST API with the Airflow UI. 
You do not need to install airflow sdk for python, just keep the python scripts for dag in the `dags` folder And refresh 
the Airflow UI to see the new DAGs.

- Check docker and docker-compose are installed
  - `docker --version`
  - `docker-compose --version`
  - If not installed follow instructions on Docker official website
- To install Airflow on Docker, follow [airflow link](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) and [youtube link](https://www.youtube.com/watch?v=J6azvFhndLg&list=PLwFJcsJ61oujAqYpMp1kdUBcPG0sE0QMT&index=3)
  - First, change directory to the "airflow_docker" dir and fetch the docker-compose.yaml file:
    ```bash
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/3.0.2/docker-compose.yaml'
    ```
  - Replace CelerayExecutor with LocalExecutor and other entris for Celery and Redis
  - Create Folders: 
    ```bash 
    mkdir -p ./dags ./logs ./plugins ./config
    ```
  - Initialize the Airflow database: (The account created has the login airflow and the password airflow)
    ```bash 
    docker compose up airflow-init
    ```
  - To start the airflow server in detach mode use:
    ```bash
    docker-compose up -d
    ```
  - Airflow UI can be accessd at `0.0.0.0:8080`. Login with username: `airflow` and password: `airflow`.
  - To Shut down the Airflow server, use: (-v or --volumes flag will remove the volumes created by docker-compose)
    ```bash
    docker-compose down --volumes
    ```
    
  
### How to run multiple projects on Airflow (You need to copy the code to the dags folder to run the DAGs)
So how you will manage multiple projects on Airflow? i.e. all code will be in the same "dags" folder. all data will be in the same "data" folder, and all logs will be in the same "logs" folder.
#### Option 1: 
A single docker+ariflow setup for each project. Keep the dags in the `dags` folder.
#### Option 2: 
A single docker+ariflow setup for all projects. Keep the dags in the `dags` folder and use subfolders for each project.
#### Option 3: 
Develope codes for each project in differnet directories, keep a dag subfolder in each project directory 
and symlink the dags to the shared `dags` folder of the Airflow setup. Or monunt the dags folder in docker-compose.yaml file.
symlink is not working, you need to copy the whole directory to the dags folder of the Airflow setup.

You do similar for data to share over mountable volumes in docker-compose.yaml file.


### Python Code for Tasks will run inside Airflow Docker Container

Reference: https://www.youtube.com/watch?v=0UepvC9X4HY&list=PLwFJcsJ61oujAqYpMp1kdUBcPG0sE0QMT&index=12

The python code for the tasks will run inside the Airflow docker container. So you must installl the required python 
packages in the Airflow docker container.

- Extend the docker image
- Customize

#### 1. Extend the docker image
- First export the python packages you need in a requirements.txt file using the command:
  ```bash
   pip list --format=freeze > requirements.txt
  ```
- Write new Dockerfile in the same airflow_docker directory that extends the Airflow docker image and intalls the requirements.txt file:
- Replace the base image in docker-compose.yaml file with the new image name.
- Stop the Airflow server if it is running:
  ```bash
  docker-compose down --volumes
  ```
- Rebuild the docker image after making change in the docker-compose.yaml file:
  ```bash
  docker-compose up -d --build
  ```
- Reinitialize the airflow database if above does not work:
  ```bash
  docker compose up airflow-init
  ```
  
### Sharing data with Airflow Docker Container
You need to share the data with the Airflow docker container to run the tasks. You can do this by mounting the data folder in the docker-compose.yaml file.

You need to add a line in the `volumes` section of the `docker-compose.yaml` file to mount the data folder into a subfolder of the Airflow container, like `/opt/airflow/data/subfolder`.
```aiignore
  volumes:
    - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
    - ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
    - ${AIRFLOW_PROJ_DIR:-.}/../../raw_data:/opt/airflow/data/raw_data
    - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
    - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
    - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins

```






