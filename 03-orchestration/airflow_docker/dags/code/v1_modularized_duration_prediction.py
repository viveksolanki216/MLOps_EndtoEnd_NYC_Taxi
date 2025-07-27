
'''
This script is a script version of the jupyter notebook `V(ExpTrack)_duration_predictor.ipynb` that we had used for
experimentation. Now that we want to create a production ready pipeline, we are moving the code to a well modularized
Python Script.

We have put the code in differnt functions and kept them in utlis.py file. This will help us to reuse the code

Now we have creaeted a dag that will use this script to run the pipeline in Airflow.

Since we are running Airflow in docker we have to make sure that this script is available in the 'dags' directory of the
Airflow Docker container. We will create a symlink to the script in the dags directory.

`ln -s code ../airflow_docker/dags/code/`

If symlink does not work, you can copy the file to the dags directory
`cp -r code airflow_docker/dags`

'''

import os
import pickle
import sys
from scipy import sparse
import numpy as np
#print(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./03-orchestration/code/')
from utils import preprocess_data, load_data, experiment_regression_model, experiment_xgboost_model


# Train the model out of mlflow context and then log the model to MLflow
# Because if anything breaks we do not need to run the whole training again

if __name__ == "__main__":

    data_dir = "../../../../raw_data/"
    train_files = ["yellow_tripdata_2025-03.parquet"]
    test_files = ["yellow_tripdata_2025-04.parquet"]

    train_data, test_data = load_data(data_dir, train_files, test_files)

    X_train, y_train, data_preprocessor_obj = preprocess_data(train_data)
    X_test, y_test, _ = preprocess_data(test_data, preprocessor_obj=data_preprocessor_obj)

    # X_train and X_test are sparse matrices, where
    sparse.save_npz("X_train.npz", X_train)
    np.save("y_train.npy", y_train)

    X_train = sparse.load_npz("X_train.npz")
    y_train = np.load("y_train.npy")

    models_dir = "../../../models/"
    os.makedirs(models_dir, exist_ok=True)
    with open('{}/preprocessor_obj.pkl'.format(models_dir), 'wb') as f_out:
        pickle.dump(data_preprocessor_obj, f_out)

    #model, train_error, test_error = experiment_regression_model(
    #    X_train,
    #    y_train,
    #    X_test,
    #    y_test
    #)
    #print("Train Error:", train_error)
    #print("Test Error:", test_error)

    experiment_xgboost_model(
        X_train,
        y_train,
        X_test,
        y_test
    )

    import  numpy as np
    best_params = {'learning_rate': np.float64(0.5844929140172085), 'max_depth': np.float64(15.0),
     'min_child_weight': np.float64(1.6633475739348214), 'n_estimators': np.float64(70.0),
     'reg_alpha': np.float64(0.06469642048222317), 'reg_lambda': np.float64(0.02179333482164594)}

    integer_params = ['max_depth', 'n_estimators']

    best_params = {
        k: (
            int(v) if k in integer_params
            else float(v) if isinstance(v, np.floating)
            else v
        )
        for k, v in best_params.items()
    }
