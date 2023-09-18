
# Car Price Prediction <img src="https://cdn-icons-png.flaticon.com/512/2168/2168422.png" alt="Car Price Prediction" width="50" height="50">


Car Price Prediction is a comprehensive machine learning project that helps estimate the selling price of cars based on various features such as year, mileage, engine capacity, and more. It includes a user-friendly web application powered by Streamlit for easy access to predictions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [Dockerized Web App](#dockerized-web-app)
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
