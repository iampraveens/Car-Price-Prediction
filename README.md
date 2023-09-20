
# Car Price Prediction <img src="https://cdn-icons-png.flaticon.com/512/2168/2168422.png" alt="Car Price Prediction" width="50" height="50">


Car Price Prediction is a comprehensive machine learning project that helps estimate the selling price of cars based on various features such as year, mileage, engine capacity, and more. It includes a user-friendly web application powered by Streamlit for easy access to predictions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Code Layout](#code-layout)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Save the Model](#save-the-model)
- [Prediction](#prediction)
- [Dockerized Web App](#dockerized-web-app)
- [Experimental Tracking](#experimental-tracking)
- [License](#license)

## Overview
Car Price Prediction is a machine learning project that aims to assist users in estimating the selling price of a car. It leverages data preprocessing techniques and employs an XGBoost regression model for accurate predictions. The project is organized into several modules for easy management and scalability.

## Project Structure
The project structure is organized as follows:

- `data/`: Contains the dataset (`car_data.csv`) used for training and predictions.
- `pipelines/`: Includes ZenML pipelines for data cleaning, model training, and evaluation.
- `steps/`: Custom Python scripts for data loading, model training, and evaluation.
- `src/`: Source code files, including data cleaning strategies, model development, and utilities.
- `saved_models/`: Stores trained machine learning models.
- `utils.py`: Utility functions for model saving and loading.
- `app.py`: Streamlit-based web application for predicting car prices.
- `requirements.txt`: Python dependencies for the project.
- `Dockerfile`: Docker configuration for containerizing the web app.

## Code Layout
![car_price_layout](https://github.com/iampraveens/Car-Price-Prediction-MLOps/assets/125688218/fada7397-0d7f-4a19-9ff0-d395b3ca31be)

## Getting Started
To get started with the project, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/iampraveens/Car-Price-Prediction-MLOps.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Training the Model

```bash
python ./steps/run_pipeline.py
```
- This command will execute the data cleaning, model training, and evaluation process

## Save the Model

```bash
python ./steps/save_model.py
```

## Prediction

```bash
streamlit run app.py
```
## Experimental Tracking
```bash
https://dagshub.com/iampraveens/Car-Price-Prediction-MLOps/experiments/
```
Here I've implemented `MLFlow` to track my models on `DagsHub`. Check out using about link

## Dockerized Web App
You can also deploy the Car Price Prediction web application using Docker. Build the Docker image and run the container:
```bash
docker build -t your_docker_username/car-price-prediction-app .
```
- To build a docker image.

```bash
docker run -d -p 8501:8501 your_docker_username/car-price-prediction-app
```
- To run as a container.

Access the web app at `http://localhost:8501` or `your_ip_address:8501` in your web browser.
Else if you want to access my pre-built container, here is the code below to pull from docker hub(Public).
```bash
docker pull iampraveens/car-price-prediction-app:latest
```
## License 
This project is licensed under the MIT License - see the [License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.

This README provides an overview of the project, its structure, how to get started, how to train the model, make predictions, tracking the model and even deploy a Dockerized web app.
