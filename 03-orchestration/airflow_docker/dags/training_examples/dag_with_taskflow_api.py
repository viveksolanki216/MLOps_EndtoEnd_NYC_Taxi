from datetime import  datetime, timedelta
from airflow.decorators import dag, task


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 12, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'scheduler_interval': '@daily',  # Use a cron expression or timedelta
}

@dag(
    dag_id="testing_dag_taskflow_api",
    description="Example DAG with TaskFlow API",
    default_args=default_args,
)
def taskflow_dag():


    @task
    def get_name():
        return {'name': 'Vivek', 'lastname': 'Solanki'}  # The returned values will be pushed to XCom and can be pulled by other tasks.

    @task
    def print_message(message, name_dict):
        name, last_name = name_dict['name'], name_dict['lastname']
        print("Message from ", name, " ", last_name, ": ", message)
        # Instead of returning values, you can also use XCom to push values to be used by other tasks.
        # you can access this values in other tasks.
        return {
            'message': message,
            'from': '{} {}'.format(name, last_name)
        }

    @task
    def send_email(message_dict, to):
        from1, message = message_dict['from'],  message_dict['message']
        print("Sending email to ", to, " from ", from1, 'and message is"', message, '"')  # Placeholder for email sending logic

    name_dict = get_name()
    message_dict = print_message("Hello ", name_dict)
    send_email(message_dict, "everyone")

dag_instance = taskflow_dag()