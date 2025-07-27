
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'start_date':datetime(2025, 7, 12, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'scheduler_interval': '#daily'#timedelta(minutes=10),
}
with DAG(
    dag_id="first_dag_example",
    description="First DAG example",
    default_args=default_args,
) as dag:
    task1 = BashOperator(
        task_id='first_task',
        bash_command='echo "Hey I am task 1"',
    )

    task2 = BashOperator(
        task_id='second_task',
        bash_command='echo "Hey I am task 2, I run after task 1"',
    )
    task3 = BashOperator(
        task_id='third_task',
        bash_command='echo "Hey I am task 3, I run in parallel with task 2"',
    )

    task1.set_downstream(task2)
    task1.set_downstream(task3)