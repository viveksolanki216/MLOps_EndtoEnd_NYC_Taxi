

FROM python:3.12-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src/predict.py .
COPY ./models/data_preprocessor_obj.pkl .
COPY ./models/XGBoost_Best_Model.pkl .

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]