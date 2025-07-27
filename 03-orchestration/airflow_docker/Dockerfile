
# To isntall all the depndencie to run python code in Airflow Docker container
# we need to extend the base Airflow image and install the required packages.
# and then replace the base image with the image generated here into the docker-compose file

# Use the official Airflow image as the base image
FROM apache/airflow:3.0.2

# swirch to root user to install packages
USER root

# Install system dependencies, I think this step is required and need to isntall gcc.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    libpq-dev build-essential && \
    apt-get clean

# Switch back to the airflow user
USER airflow


COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /requirements.txt