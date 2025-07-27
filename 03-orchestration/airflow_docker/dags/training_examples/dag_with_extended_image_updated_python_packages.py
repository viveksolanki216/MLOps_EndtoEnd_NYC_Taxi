from airflow import  DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

from mlflow import xgboost


def print_message(message, ti):
    import mlflow
    from sklearn.feature_extraction import DictVectorizer
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import json
    import pickle
    #print(sklearn.__version__)
    #print(mlflow.__version__)
    #print(xgboost.__version__)


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 12, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'scheduler_interval': '@daily',  # Use a cron expression or timedelta
}

with DAG(
    dag_id="testing_dag_for_new_extended_image_python_packages",
    description="Example DAG with Python Operators",
    default_args=default_args,
) as dag:
    task1 = PythonOperator(
        task_id='get_message_task',
        python_callable=print_message,
        op_kwargs={'message': 'Hello from task 1'},
    )
task1