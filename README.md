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
* Usamos el comando `create_api` y después `create_docker`.
* Ejecutamos el contenedor con: `docker build .`. Asumimos que estamos en la misma carpeta.
* Podemos revisar la imagen con `docker images -a`. Con esto podemos ver las imágenes que tenemos en nuestro sistema.
* Ejecutamos el contenedor con `docker run `
