
import os
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

app = Flask("duration-prediction")


#cwd = "/Users/vss/Personal/Git/MLOps_EndtoEnd_NYC_Taxi/"
#artifact_dir = f"{cwd}/03-orchestration/airflow_docker/data/mlflow_artifacts/models/"
artifact_dir = "./"
model_location = f"{artifact_dir}XGBoost_Best_Model.pkl"#.format(artifact_dir)
preprocessor_location = f"{artifact_dir}data_preprocessor_obj.pkl"#.#format(artifact_dir)

def load_models(model_location, preprocessor_location):
    """
    Load the preprocessor and model objects from the specified locations.
    :param model_location:
    :param preprocessor_location:
    :return:
    """
    with open(preprocessor_location, "rb") as f:
        data_preprocessor_obj = pickle.load(f)

    with open(model_location, "rb") as f:
        model_obj = pickle.load(f)

    return data_preprocessor_obj, model_obj


def prepare_featrures(request):
    """
    Prepare the features for prediction.
    """
    request = {
        'PU_DO': f"{request['PULocationID']}_{request['DOLocationID']}",
        'trip_distance': request['trip_distance']
    }
    print(request)
    return request


def transform_features(data_preprocessor_obj, request):
    """
    Transform the input request using the preprocessor object.
    """
    X = data_preprocessor_obj.transform([request])
    return X


def predict(model_obj, X):
    X = xgb.DMatrix(X)
    pred = model_obj.predict(X)
    return pred


def predict_duration(request):
    """
    Predict the duration based on the input request.
    """
    data_preprocessor_obj, model_obj = load_models(model_location, preprocessor_location)

    request = prepare_featrures(request)

    X = transform_features(data_preprocessor_obj, request)

    pred = predict(model_obj, X)

    return pred  # Return the first prediction value

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    preds = predict_duration(ride)

    result = {'duration': float(preds[0])}  # Assuming the prediction is a single value
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)