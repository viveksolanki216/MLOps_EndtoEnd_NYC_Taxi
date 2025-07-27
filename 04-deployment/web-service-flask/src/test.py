

import requests
request = {
    'PULocationID': 100,
    'DOLocationID': 100,
    'trip_distance': 0.36
}
url = " http://127.0.0.1:9696/predict"

response = requests.post(url, json=request)
print(response)
print(response.json())


















#import sys
#sys.path.append("04-deployment/web-service-flaskapi/")
#import predict

#request = {
#    'PULocationID': 100,
#    'DOLocationID': 100,
#    'trip_distance': .35
#}

#preds = predict.predict_duration(request)
#print(preds)   A