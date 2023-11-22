# pycaret-tutorial
Repository containing some examples of pycaret library. The objetive is to be a tutorial for ML Deployment with PyCaret.

## Preparation

1. `pip install pycaret`
2. `pip install scikit-learn-intelex`
3. `pip install pycaret[analysis]` for dashboards
4. `pip install pycaret[mlops]` for gradio and mlflow

## About PyCaret

* PyCaret is an open-source library that works as a wrpapper of other Python libraries for Machine Learning. 
* It can be used for:
    * Classification
    * Regression
    * Anomaly detection
    * Clustering
    * Time Series
* It also has capabilities for experiment tracking (it can use MLFlow for that) and for deployment.

## Deploying with PyCaret

* PyCaret can help us to deploy models with Docker. 
* We use `create_api` then `create_docker`.
* Create container with `docker build .` or `docker image build -f "Dockerfile" -t ml_api:v3 .`
* Run container with `docker run -p 8000:8000/tcp ml_api:v2`

## Fixes

There are somethings that have to be fixed before using the ml_api.py with Docker:
* Ensure to use pydantic < 2.0.
* Modify the host IP in the script to 0.0.0.0.
* Map the exposed port.
