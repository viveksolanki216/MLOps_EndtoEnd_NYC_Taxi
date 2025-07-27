


import os
import pickle
import xgboost as xgb

cwd = os.getcwd()

artifact_dir = f"{cwd}/03-orchestration/airflow_docker/data/mlflow_artifacts/models/"
model_location = f"{artifact_dir}XGBoost_Best_Model.pkl"#.format(artifact_dir)
preprocessor_location = f"{artifact_dir}data_preprocessor_obj.pkl"#.#format(artifact_dir)

with open(preprocessor_location, "rb") as f:
    data_preprocessor_obj = pickle.load(f)

with open(model_location, "rb") as f:
    model_obj = pickle.load(f)


request = {
    'PULocationID': 100,
    'DOLocationID': 200,
    'trip_distance': 20
}

request = {
    'PU_DO': f'{request['PULocationID']}_{request['DOLocationID']}',
    'trip_distance': 20
}
print(request)

X = data_preprocessor_obj.transform([request])


X = xgb.DMatrix(X)
model_obj.predict(X)