
import os
import pickle
import logging
from typing import Dict, Any, Tuple, Union
import xgboost as xgb
import numpy as np
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask("duration-prediction")

# Global variables to store loaded models
_data_preprocessor_obj = None
_model_obj = None
_models_loaded = False


#cwd = "/Users/vss/Personal/Git/MLOps_EndtoEnd_NYC_Taxi/"
#artifact_dir = f"{cwd}/03-orchestration/airflow_docker/data/mlflow_artifacts/models/"
artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "./")
model_location = f"{artifact_dir}XGBoost_Best_Model.pkl"
preprocessor_location = f"{artifact_dir}data_preprocessor_obj.pkl"

def load_models(model_location: str, preprocessor_location: str) -> Tuple[Any, Any]:
    """
    Load the preprocessor and model objects from the specified locations.
    :param model_location: Path to the model file
    :param preprocessor_location: Path to the preprocessor file
    :return: Tuple of (preprocessor, model) objects
    :raises: FileNotFoundError if model files don't exist
    :raises: Exception if models can't be loaded
    """
    try:
        if not os.path.exists(preprocessor_location):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_location}")
        
        if not os.path.exists(model_location):
            raise FileNotFoundError(f"Model file not found: {model_location}")
        
        logger.info(f"Loading preprocessor from: {preprocessor_location}")
        with open(preprocessor_location, "rb") as f:
            data_preprocessor_obj = pickle.load(f)

        logger.info(f"Loading model from: {model_location}")
        with open(model_location, "rb") as f:
            model_obj = pickle.load(f)

        logger.info("Models loaded successfully")
        return data_preprocessor_obj, model_obj
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def initialize_models() -> None:
    """
    Initialize models at application startup.
    This function loads models once and stores them in global variables.
    """
    global _data_preprocessor_obj, _model_obj, _models_loaded
    
    try:
        if not _models_loaded:
            logger.info("Initializing models at startup...")
            _data_preprocessor_obj, _model_obj = load_models(model_location, preprocessor_location)
            _models_loaded = True
            logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        _models_loaded = False
        raise


def get_models() -> Tuple[Any, Any]:
    """
    Get the loaded models. If not loaded, initialize them.
    :return: Tuple of (preprocessor, model) objects
    :raises: Exception if models can't be loaded
    """
    global _data_preprocessor_obj, _model_obj, _models_loaded
    
    if not _models_loaded:
        initialize_models()
    
    return _data_preprocessor_obj, _model_obj


def prepare_features(request: Dict[str, Any]) -> Dict[str, Union[str, float]]:
    """
    Prepare the features for prediction.
    :param request: Dictionary containing input features
    :return: Dictionary with prepared features
    :raises: KeyError if required fields are missing
    """
    try:
        if not request:
            raise ValueError("Request data cannot be empty")
        
        # Validate required fields
        required_fields = ['PULocationID', 'DOLocationID', 'trip_distance']
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            raise KeyError(f"Missing required fields: {missing_fields}")
        
        prepared_request = {
            'PU_DO': f"{request['PULocationID']}_{request['DOLocationID']}",
            'trip_distance': float(request['trip_distance'])
        }
        logger.info(f"Prepared features: {prepared_request}")
        return prepared_request
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


def transform_features(data_preprocessor_obj: Any, request: Dict[str, Union[str, float]]) -> Any:
    """
    Transform the input request using the preprocessor object.
    :param data_preprocessor_obj: Fitted preprocessor object
    :param request: Dictionary with prepared features
    :return: Transformed feature matrix
    :raises: Exception if transformation fails
    """
    try:
        X = data_preprocessor_obj.transform([request])
        logger.info(f"Features transformed successfully, shape: {X.shape}")
        return X
    except Exception as e:
        logger.error(f"Error transforming features: {str(e)}")
        raise


def predict(model_obj: Any, X: Any) -> np.ndarray:
    """
    Make prediction using the model.
    :param model_obj: Trained model object
    :param X: Transformed features
    :return: Prediction array
    :raises: Exception if prediction fails
    """
    try:
        X = xgb.DMatrix(X)
        pred = model_obj.predict(X)
        logger.info(f"Prediction completed successfully")
        return pred
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def predict_duration(request: Dict[str, Any]) -> np.ndarray:
    """
    Predict the duration based on the input request.
    :param request: Dictionary containing trip information
    :return: Prediction array
    :raises: Exception if any step fails
    """
    try:
        # Use cached models instead of loading on every request
        data_preprocessor_obj, model_obj = get_models()
        
        prepared_request = prepare_features(request)
        
        X = transform_features(data_preprocessor_obj, prepared_request)
        
        pred = predict(model_obj, X)
        
        return pred
    except Exception as e:
        logger.error(f"Error in predict_duration: {str(e)}")
        raise

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    API endpoint for trip duration prediction.
    Expected JSON payload: {
        "PULocationID": int,
        "DOLocationID": int, 
        "trip_distance": float
    }
    Returns: {"duration": float} or error message
    """
    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        ride = request.get_json()
        
        if not ride:
            return jsonify({"error": "Empty request body"}), 400
        
        logger.info(f"Received prediction request: {ride}")
        
        preds = predict_duration(ride)
        
        result = {'duration': float(preds[0])}
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    except ValueError as e:
        error_msg = f"Invalid input value: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    except FileNotFoundError as e:
        error_msg = f"Model files not found: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": "Service temporarily unavailable"}), 503
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify service status.
    """
    try:
        # Check if model files exist
        models_exist = (
            os.path.exists(model_location) and 
            os.path.exists(preprocessor_location)
        )
        
        status_info = {
            "status": "healthy" if models_exist and _models_loaded else "unhealthy",
            "models_loaded": _models_loaded,
            "model_location": model_location,
            "preprocessor_location": preprocessor_location,
            "model_files_exist": models_exist
        }
        
        if models_exist and _models_loaded:
            return jsonify(status_info), 200
        else:
            status_info["error"] = "Models not loaded or files not found"
            return jsonify(status_info), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

if __name__ == "__main__":
    # Use environment variables for configuration
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "9696"))
    
    logger.info(f"Starting Flask app on {host}:{port}, debug={debug_mode}")
    
    # Initialize models at startup for better performance
    try:
        initialize_models()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models at startup: {str(e)}")
        logger.warning("Application will start but models will be loaded on first request")
    
    app.run(debug=debug_mode, host=host, port=port)