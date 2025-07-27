
'''
This script is a script version of the jupyter notebook `V(ExpTrack)_duration_predictor.ipynb` that we had used for
experimentation. Now that we want to create a production ready pipeline, we are moving the code to a well modularized
Python Script.

Then in second version we will create a DAG that will use this script to run the pipeline in Airflow.

Then in third version we will introduce MLflow to track the experiments and log the model.

'''

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll import scope
from enum import Enum


class Transform_Fun_Options(str, Enum):
    LOG = 'log'
    SQRT = 'sqrt'




def calculate_target_variable(
        data: pd.DataFrame
) -> pd.DataFrame:
    '''
    calcualte the trip duration from pickup time and drop time
    '''
    data['trip_duration'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
    data['trip_duration'] = data['trip_duration'].dt.total_seconds()/60#.apply(lambda diff: diff.total_seconds() / 60)
    print(data['trip_duration'].describe())
    print(data['trip_duration'].head())
    return data


def filter_outliers(
        data: pd.DataFrame
) -> pd.DataFrame:
    flag1 = (data['trip_duration'] >= 1) & (data['trip_duration'] <= 60)
    flag2 = (data['trip_distance'] >= 0.2) & (data['trip_distance'] <= 100)
    print(flag1.sum())
    print((flag1 & flag2).sum())
    data = data[flag1 & flag2].reset_index(drop=True)
    return data


def transform_numeric_features(
        data: pd.DataFrame,
        features: list,
        transformation_fun: Transform_Fun_Options
):
    if transformation_fun == Transform_Fun_Options.LOG:
        data[features] = np.log(data[features] + 1)
    return data



def load_parquet_data(
        file: str
) -> pd.DataFrame:
    '''
    Loads parquet data
    '''
    print("Loading data from: ", file)
    data = pd.read_parquet(file)
    print("\n======= SIZE =========")
    print(data.shape)
    print("\n======= DTYPES =========")
    print(data.dtypes)
    print("\n======= SUMMARY =========")
    print(data.describe())

    return data

def load_data(
        data_dir: str,
        files_list: list
):
    '''
    Loads the data from the given directory and files list.
    :param data_dir:
    :param files_list:
    :return:
    '''

    data = pd.concat([load_parquet_data('{}{}'.format(data_dir,file)) for file in files_list], axis=0)
    return data

def preprocess_data(
        data: pd.DataFrame,
        preprocessor_obj: DictVectorizer=None,
        numerical_features: list = ['trip_distance'],
        categorical_features: list = ['PULocationID', 'DOLocationID']
) -> tuple:

    '''
    Preprocess the data for training or testing.
    Args:
        data (pd.DataFrame): The input data to preprocess.
        preprocessor_obj (DictVectorizer, optional): Preprocessor object for transforming categorical features.
        numerical_features (list, optional): List of numerical features to include.
        categorical_features (list, optional): List of categorical features to include.
    '''
    # Calcualte Target i.e. Trip duration
    data = calculate_target_variable(data)
    print("Target calulated")

    # Create PU_DO feature
    if len(set(['PULocationID', 'DOLocationID']).intersection(categorical_features))==2:
        data['PU_DO'] = data['PULocationID'].astype('str') + '_' + data['DOLocationID'].astype('str')

    print("PU_DO feature created")

    # Filter outliers
    #data_all = data.copy()
    data = filter_outliers(data)
    print(data.shape)
    print(data['trip_duration'].describe())


    # Target and Features
    categorical_features = ['PU_DO']#['PULocationID', 'DOLocationID']
    features = numerical_features + categorical_features
    target = 'trip_duration'

    # Transform features
    #features_to_transform = numerical_features + [target]
    #data = transform_numeric_features(data, features_to_transform, Transform_Fun_Options.LOG)
    #test_data = transform_numeric_features(test_data, features_to_transform, Transform_Fun_Options.LOG)
    #test_data = transform_numeric_features(test_data, features_to_transform, Transform_Fun_Options.LOG)

    # Change all categorical variables to strings
    data[categorical_features] = data[categorical_features].astype('str')
    #data_all[categorical_features] = data_all[categorical_features].astype('str')


    y = data[target].values

    ## Preparing the X, y for training and testing
    if not preprocessor_obj: # Training Preprocessing
        print("Preprocessing data for training")
        # There are many rare PU_DO combinations, that will creates around 38K unique combinations, so we will replace the rare with 'other'
        # Make this rare trip combination as 'other'
        Freq = data['PU_DO'].value_counts().reset_index()
        rare_trips = set(Freq.loc[Freq['count'] <= 5, 'PU_DO'])
        data['PU_DO'] = np.where(data['PU_DO'].isin(rare_trips), 'Other', data['PU_DO'])
        print(data['PU_DO'].nunique())

        # Change dataframe to dict format for one hot encoding for categorical variables
        data_dict = data[features].to_dict(orient='records')
        print(data_dict[0])

        print("Creating new preprocessor object to fit and transform data")
        preprocessor_obj = DictVectorizer()
        X = preprocessor_obj.fit_transform(data_dict)
        print(X.shape)
    else: # Testing Preprocessing
        # Change dataframe to dict format for one hot encoding for categorical variables
        data_dict = data[features].to_dict(orient='records')
        print(data_dict[0])
        print("Using existing preprocessor object to transform data")
        X = preprocessor_obj.transform(data_dict)
        print(X.shape)

    return X, y, preprocessor_obj


def experiment_regression_model(
        X_train,
        y_train,
        X_test,
        y_test
):

    # Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    #print(model.coef_, model.intercept_)
    print("Model is Trained")

    # Make Predictions
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    # Score Predictions
    train_error = root_mean_squared_error(y_train, y_train_preds)
    test_error = root_mean_squared_error(y_test, y_test_preds)
    print("Train Error:", train_error)
    print("Test Error:", test_error)

    params = {}
    return params, train_error, test_error

# Objective function for hyperparameter tuning using Hyperopt
def objective(params, train, valid, y_train, y_test, num_boost_round=2, early_stopping_rounds=1):
    '''
    Objective function for hyperparameter tuning
    '''
    print(params)
    # Train the model
    model = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=num_boost_round,
        evals=[(valid, 'eval')],
        early_stopping_rounds=early_stopping_rounds
    )

    # Make Predictions
    y_train_preds = model.predict(train)
    y_test_preds = model.predict(valid)

    # Score Predictions
    train_error = root_mean_squared_error(y_train, y_train_preds)
    test_error = root_mean_squared_error(y_test, y_test_preds)
    print("Train Error:", train_error)
    print("Test Error:", test_error)

    # objective function tries to serialize the model, and instead of sending modela again and again I think best
    # to retrain the model with the best parameters and return the test error
    return {'loss': test_error, 'status': STATUS_OK, 'train_loss': train_error, 'test_loss': test_error, 'params': params}

def experiment_xgboost_model(
        X_train,
        y_train,
        X_test,
        y_test,
        n_experiments=2,
        num_boost_round=2,
        early_stopping_rounds=1
):
    '''
    Experiment with XGBoost model
    '''
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)

    integer_params = ['max_depth', 'n_estimators']
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 3)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -4, 0),
        'reg_lambda': hp.loguniform('reg_lambda', -4, 0),
        'min_child_weight': hp.loguniform('min_child_weight', 0, 3),
        'objective': 'reg:squarederror',
        'seed': 42,
        'n_jobs': 1,  # Use all available cores
    }

    objective_fun = lambda params: objective(
        params=params,
        train=train,
        valid=valid,
        y_train=y_train,
        y_test=y_test,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )
    trials = Trials() # Stores run information about the trials, i.e. explore results like trails.trials[0]['result']['loss']
    best_params = fmin(
        fn=objective_fun,  # Objective function to minimize
        space=search_space,
        algo=tpe.suggest,  # Algorithm for space searching (smarter then random seach as in grid search)
        max_evals=n_experiments,  # Maximum number of combinatios of hyperparameters to try
        trials=trials
        # Stores information about the trials, i.e. explore results like trails.trials[0]['result']['loss']
    )

    return trials


def create_final_model(
        X_train, y_train, X_test, y_test, best_params,
        num_boost_round=2,
        early_stopping_rounds=1
):


    print("Best params found", best_params)
    #best_params = {
    #    k: (
    #        int(v) if k in integer_params
    #        else float(v) if isinstance(v, np.floating)
    #        else v
    #    )
    #    for k, v in best_params.items()
    #}

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_test, label=y_test)

    # Train the model with best parameters
    print("Format change best params", best_params)
    model = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=num_boost_round,
        evals=[(valid, 'eval')],
        early_stopping_rounds=early_stopping_rounds
    )

    # Make Predictions
    y_test_preds = model.predict(valid)
    test_error = root_mean_squared_error(y_test, y_test_preds)
    y_train_preds = model.predict(train)
    train_error = root_mean_squared_error(y_train, y_train_preds)
    print("Test Error of the best model:", test_error)

    return model, test_error, train_error