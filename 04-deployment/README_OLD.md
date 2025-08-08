
# Deploying Models

- Create a predict.py file that contains the model loading and prediction logic
- Use Flask to create a web service that can handle HTTP requests for predictions
- Use a WSGI server like Gunicorn or uWSGI to serve the Flask app in production
- create a requirements.txt file with all the dependencies `pip list --format=freeze > requirements.txt`
- Use Docker to containerize the application for easy deployment and scaling
- Copy models in the build context i.e. same directory or in subdirectory as of dockerfile.
- `docker build . -t trip_duration_predictor_service:v1`
## Over Flask

# Flask
 - Lightweight, easy to use python framework for building web applications and APIs
 - Let's easily covnert Python code, models, data pipelines into a web service that other code can interact with over HTTP
 - Great for quick prototyping, low traffic deployment, demo apps, 
 - Stable and widely used for production APIs at many small-scale companies

## Key Features
- Low Code to create APIs
- Flexibility: Works with any ML/DL framework (TensorFlow, PyTorch, Scikit-learn, etc.)
- Ecosystem: Easily adds api docs, auth, monitoring

## Use-Cases
- Serving Models as API over HTTP i.e. "http://localhost:5000/predict"
- Batch Scoring Triggers
- MLOps Metrics
- Prototyping Real Data

## How flask should deployed at production
- Use it with gunicorn or uWSGI to handle multiple requests

### gunicorn
- > pip isntall gunicorn
- Dont run flask app directly in production
- Run with gunicorn to handle multiple requests
- > gunicorn --bind=0.0.0.0:9696 predict:app
