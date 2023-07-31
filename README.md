<div align='center'>

## Temperature Forecasting
</div>

This repository contains a temperature forecasting project that aims to predict future temperature values using different machine learning models. The main goal is to compare the performance of three models: linear regression, XGBoost, and NeuralProphet, to identify the most suitable model for the temperature forecasting task.

### Overview

The project is focused on exploring and comparing different machine learning approaches for temperature forecasting. The dataset used in this project includes historical temperature data, and the objective is to build predictive models that can forecast temperature values for future time points.

## Data Source

The data used in this project is publicly available and can be obtained from the [Météo-France Public Data Portal](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32). It provides historical records of several atmospheric parameters, including temperature, from various locations, which will be used to train and test forecast models.


### Models

The following machine learning models will be evaluated in the project:

1. **Linear Regression**: This is a simple and interpretable model that assumes a linear relationship between the input features and the target variable.

2. **XGBoost**: XGBoost is a popular gradient boosting model based on decision trees which is known for its high predictive power and efficiency in handling complex data.

3. **NeuralProphet**: NeuralProphet is a model that combines the power of neural networks with the interpretability of time series models, making it well-suited for time series forecasting tasks.

### Project Structure

The repository is organized as follows:

- `data/`: This directory contains the dataset used for training and testing the models.

- `model/`: An empty directory to contain the trained NeuralProphet model.

- `main.py`: This file contains the lines of code used to train the models and display the predictions on a graph.

