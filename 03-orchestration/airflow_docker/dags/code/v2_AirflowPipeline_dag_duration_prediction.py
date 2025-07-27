'''
This dag uses decorators to define the tasks and the DAG structure.
'''
import gc
import os
import pickle
from datetime import datetime, timedelta
import numpy as np
from requests.utils import set_environ
from scipy import sparse
from airflow.decorators import dag, task
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_data, load_data, experiment_regression_model, experiment_xgboost_model, create_final_model
import mlflow
# Setting Up MLflow tracking URI and set a project as a single experiment

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 7, 12, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'scheduler_interval': '@daily',  # Use a cron expression or timedelta
}

mlflow.set_tracking_uri(f"http://host.docker.internal:5000")
mlflow.set_experiment("NYC Yellow Cab Trip Duration Prediction")

@dag(
    dag_id="DAG_Duration_Prediction",
    description="DAG for Duration Prediction Pipeline",
    default_args=default_args,
)
def duration_prediction():

    # You need to return a single object from the DAG function and it should be passed to the next task as is.
    # You can take each element of the object only inside the task function. not outside when defining the dependencies.

    #

    def get_paths(docker_container_data_path):
        # Set input data directory
        data_dir = f'{docker_container_data_path}/raw_data/'

        # Crate a location to store the models and pre-processor objects
        models_dir = f"{docker_container_data_path}/mlflow_artifacts/models/"
        os.makedirs(models_dir, exist_ok=True)

        # Create a location and dictionary to store processed data locations
        processed_data_dir = f"{docker_container_data_path}/processed_data/"
        os.makedirs(processed_data_dir, exist_ok=True)

        processed_data_location_dict = {
            'X_train': f"{processed_data_dir}/X_train.npz",
            'y_train': f"{processed_data_dir}/y_train.npy",
            'X_test': f"{processed_data_dir}/X_test.npz",
            'y_test': f"{processed_data_dir}/y_test.npy"
        }

        location_params = {
            'Docker Data Path': docker_container_data_path,
            'Raw Data Dir Path': data_dir,
            'Models Dir Path': models_dir,
            'Processed Data Dir Path': processed_data_dir,
            'Processed Data Files Paths': processed_data_location_dict
        }
        return location_params

    @task
    def set_environment():
        # Docker Container Data dir is mounted on the local data path
        docker_container_data_path = '/opt/airflow/data'
        local_data_path = '//03-orchestration/airflow_docker/data'

        env_info = get_paths(docker_container_data_path)
        env_info['Local Data Path'] = local_data_path
        # Parent Run + Nested Runs
        # A parent run for each pipeline and child runs for each task/experiment for hyperparameter tuning
        with mlflow.start_run(run_name="Pipeline Run") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.end_run()
        print("Started MLFLow Run")


        env_info['MLFlow Parent Run ID'] = parent_run_id
        return env_info

    @task
    def load_and_preprocess_train_data(train_files: list, env_info: dict):

        data_dir = env_info['Raw Data Dir Path']
        # first load the data
        train_data = load_data(data_dir, train_files)

        # preprocess the data
        X_train, y_train, data_preprocessor_obj = preprocess_data(train_data)
        sparse.save_npz(env_info['Processed Data Files Paths']['X_train'], X_train)
        np.save(env_info['Processed Data Files Paths']['y_train'], y_train)
        del X_train, y_train
        gc.collect()

        # Store the preprocessor object for later use
        preprocessor_location = '{}/data_preprocessor_obj.pkl'.format(env_info['Models Dir Path'])
        with open('{}/data_preprocessor_obj.pkl'.format(env_info['Models Dir Path']), 'wb') as f_out:
            pickle.dump(data_preprocessor_obj, f_out)

        env_info['Data Preprocessor Object'] = preprocessor_location

        print(preprocessor_location)
        # Track the task using MLflow

        with mlflow.start_run(run_id=env_info['MLFlow Parent Run ID']) as parent_run:
            with mlflow.start_run(run_name="Loading and Preprocessing Train Data", nested=True) as child_run:
                mlflow.log_param("Input Data Dir", env_info['Raw Data Dir Path'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))
                mlflow.log_param("Input Train Files", train_files)
                mlflow.log_param("Output Data Preprocessor Object", env_info['Data Preprocessor Object'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))
                mlflow.log_param("Output Processsed Data Path", env_info['Processed Data Dir Path'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))
                #print("---", env_info['Data Preprocessor Object'])
                #mlflow.log_artifact(env_info['Data Preprocessor Object'], artifact_path="preprocessor_obj")
                #print("++++++++", env_info['Data Preprocessor Object'])
            mlflow.end_run()
        return env_info


    @task
    def load_and_preprocess_test_data(test_files: list, env_info: dict):
        data_dir = env_info['Raw Data Dir Path']

        # first load the data
        test_data = load_data(data_dir, test_files)

        # First load the obj
        with open(env_info['Data Preprocessor Object'], 'rb') as f_in:
            data_preprocessor_obj = pickle.load(f_in)
        # preprocess the data
        X_test, y_test, _ = preprocess_data(test_data, preprocessor_obj=data_preprocessor_obj)
        sparse.save_npz(env_info['Processed Data Files Paths']['X_test'], X_test)
        np.save(env_info['Processed Data Files Paths']['y_test'], y_test)
        del X_test, y_test
        gc.collect()

        # Track the task using MLflow
        with mlflow.start_run(run_id=env_info['MLFlow Parent Run ID']) as parent_run:
            with mlflow.start_run(run_name="Loading and Preprocessing Test Data", nested=True) as child_run:
                mlflow.log_param("Input Data Dir", env_info['Raw Data Dir Path'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))
                mlflow.log_param("Inpu Test Files", test_files)
                mlflow.log_param("Input Data Preprocessor Object", env_info['Data Preprocessor Object'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))
                mlflow.log_param("Output Processsed Data Path", env_info['Processed Data Dir Path'].replace(env_info['Docker Data Path'], env_info['Local Data Path']))

        return env_info


    def load_processed_data(env_info: dict):
        """
        This function will load the processed data from the environment info.
        """
        processed_data_location_dict = env_info['Processed Data Files Paths']
        X_train = sparse.load_npz(processed_data_location_dict['X_train'])
        y_train = np.load(processed_data_location_dict['y_train'],)
        X_test = sparse.load_npz(processed_data_location_dict['X_test'])
        y_test = np.load(processed_data_location_dict['y_test'])

        return X_train, y_train, X_test, y_test

    @task
    def experiments_model_logistic_regression(env_info: dict):
        """
        This function will train the model and return the model and the errors.
        """

        # Load the data
        X_train, y_train, X_test, y_test = load_processed_data(env_info)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        print("Running an experiment with Logistic Regression model")
        params, train_error, test_error = experiment_regression_model(X_train, y_train, X_test, y_test)
        print("Best LR Model", train_error, test_error, params)

        model_experiment_info = {
            'name':'Logistic Regression', 'params': params, 'train_error': train_error, 'test_error': test_error
        }

        # Track the task using MLflow
        with mlflow.start_run(run_id=env_info['MLFlow Parent Run ID']) as parent_run:
            with mlflow.start_run(run_name="Logistic Regression Experiments", nested=True) as child_run:
                mlflow.set_tag("Model_Type","Logistic Regression")
                mlflow.log_param("Input Data Path",
                                 env_info['Processed Data Dir Path'].replace(env_info['Docker Data Path'],
                                                                             env_info['Local Data Path']))
                mlflow.log_params(params)
                mlflow.log_metric("Train Error", train_error)
                mlflow.log_metric("Test Error", test_error)

        return model_experiment_info

    @task
    def experiments_model_xgboost(env_info: dict, n_experiments: int = 20,
        num_boost_round: int = 10,
        early_stopping_rounds: int = 5,
        ):
        """
        This function will train the model and return the model and the errors.
        """

        # Load the data
        X_train, y_train, X_test, y_test = load_processed_data(env_info)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        print("Running an experiment with XGBoost model")
        trials = experiment_xgboost_model(
            X_train, y_train, X_test, y_test, n_experiments, num_boost_round, early_stopping_rounds
        )

        best_trials = trials.best_trial
        best_params = best_trials['result']['params']
        best_params['num_boost_round'] = num_boost_round
        best_params['early_stopping_rounds'] = early_stopping_rounds
        model_experiment_info = {
            'name': 'XGBoost',
            'params': best_params,
            'train_error': best_trials['result']['train_loss'],
            'test_error': best_trials['result']['test_loss']
        }
        print(model_experiment_info)

        with mlflow.start_run(run_id=env_info['MLFlow Parent Run ID']) as parent_run:
            for i, trial in enumerate(trials.trials):
                result = trial['result']
                with mlflow.start_run(run_name="XGBoost Regression Experiments", nested=True) as child_run:
                    mlflow.set_tag("Model_Type","XGBoost")
                    mlflow.log_param("Input Data Path",
                                     env_info['Processed Data Dir Path'].replace(env_info['Docker Data Path'],
                                                                                 env_info['Local Data Path']))

                    mlflow.log_params(result['params'])
                    mlflow.log_param("num_boost_round", num_boost_round)
                    mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
                    mlflow.log_metric("Train Error", result['train_loss'])
                    mlflow.log_metric("Test Error", result['test_loss'])

        return model_experiment_info


    @task
    def train_best_model(env_info: dict, model2_experiment_info: dict):
        """
        This function will train the model and return the model and the errors.
        """
        # Load the data
        X_train, y_train, X_test, y_test = load_processed_data(env_info)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Load the best model parameters
        best_params = model2_experiment_info['params']
        num_boost_round=best_params.pop('num_boost_round')
        early_stopping_rounds=best_params.pop('early_stopping_rounds')

        model, test_error, train_error = create_final_model(
            X_train, y_train, X_test, y_test, best_params, num_boost_round, early_stopping_rounds
        )
        print("Test Error of the model: ", test_error, "Should be equal to the experiment", model2_experiment_info['test_error'])

        # save model in the models directory
        model_path = f"{env_info['Models Dir Path']}/XGBoost_Best_Model.pkl"
        with open(model_path, 'wb') as f_out:
            pickle.dump(model, f_out)
        print("Saved the model to: ", model_path)

        with mlflow.start_run(run_id=env_info['MLFlow Parent Run ID']) as parent_run:
            with mlflow.start_run(run_name="Best Model", nested=True) as child_run:
                mlflow.set_tag("Model_Type",model2_experiment_info['name'])
                mlflow.set_tag("Best Model", 1)
                mlflow.log_params(best_params)
                mlflow.log_param("num_boost_round", num_boost_round)
                mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
                mlflow.log_metric("Train Error", train_error)
                mlflow.log_metric("Test Error", test_error)
                #mlflow.xgboost.log_model(model, name="XGBoost_Best_Model")

        return ""

    # -------------------------------------------------------------------------------
    # Define the task dependencies
    # ---------------------------------------------------------------------------------
    env_info = set_environment()
    env_info_task_train = load_and_preprocess_train_data(
        ['yellow_tripdata_2025-03.parquet'], env_info
    )
    env_info_task_test = load_and_preprocess_test_data(
        ['yellow_tripdata_2025-04.parquet'], env_info_task_train
    )
    #model1_experiment_info = experiments_model_logistic_regression(env_info_task_test)
    model2_experiment_info = experiments_model_xgboost(env_info_task_test)
    train_best_model = train_best_model(env_info_task_test, model2_experiment_info)


# -------------------------------------------------------------------------------
# Launch the DAG
# ---------------------------------------------------------------------------------
duration_prediction_dag = duration_prediction()