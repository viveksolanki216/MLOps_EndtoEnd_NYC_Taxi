from airflow import  DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

def print_message(message, ti):
    name, last_name = ti.xcom_pull(task_ids='get_name_task') # Get return value from the "get_name_task" using xcom
    print("Message from ",name, " ", last_name,": ", message)
    # Instead of returning values, you can also use XCom to push values to be used by other tasks.
    # you can access this values in other tasks.
    ti.xcom_push(key='message', value=message)  # Push a message to XCom
    ti.xcom_push(key='from', value='{} {}'.format(name, last_name))

def get_name():
    return 'Vivek', 'Solanki' # The returned values will be pushed to XCom and can be pulled by other tasks.

def send_email(to, ti):
    from1 = ti.xcom_pull(task_ids="get_message_task", key="from")  # Pull the message from the previous task
    message = ti.xcom_pull(task_ids="get_message_task", key="message")  # Pull the message from the previous task
    print("Sending email to ", to, " from ", from1, 'and message is"', message, '"')  # Placeholder for email sending logic

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 12, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'scheduler_interval': '@daily',  # Use a cron expression or timedelta
}

with DAG(
    dag_id="testing_dag_for_python_operators",
    description="Example DAG with Python Operators",
    default_args=default_args,
) as dag:
    task1 = PythonOperator(
        task_id='get_message_task',
        python_callable=print_message,
        op_kwargs={'message': 'Hello from task 1'},
    )

    task2 = PythonOperator(
        task_id="get_name_task",
        python_callable=get_name
    )

    task3 = PythonOperator(
        task_id="send_email",
        python_callable=send_email,
        op_kwargs={'to': 'everyone'}  # Pass the task instance to access XCom
    )
    task2 >> task1 >> task3  # Set task dependencies