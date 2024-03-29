<div align='center'>

## Temperature Forecasting

<img src="static/temperature_icon.png" alt="Temperature Icon" width="200px" height="190px">

</div>

<br>

### Overview

This project is focused on exploring and comparing different machine learning approaches for temperature forecasting. The dataset used in this project includes historical temperature data, and the objective is to build predictive models that can forecast temperature values for future time points.

### Data Source

The data used in this project is publicly available and can be obtained from the [Météo-France Public Data Portal](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32). It provides historical records of several atmospheric parameters, including temperature, from various locations, which will be used to train and test forecast models.

Arbitrarily, we choose station number 7005. Here is the daily evolution of the average temperature. The objective is to predict the missing part, which represents 20% of the data collected, and compare it to the actual temperatures.

<div align='center'>
<img src="static/temperature_evolution.png" alt="Temperature Evolution" width="700px" height="350px">
</div>

<br>

### Models

The following machine learning models will be evaluated in the project:

1. **Linear Regression**: This is a simple and interpretable model that assumes a linear relationship between the input features and the target variable.

2. **XGBoost**: XGBoost is a popular gradient boosting model based on decision trees which is known for its high predictive power and efficiency in handling complex data.

3. **NeuralProphet**: NeuralProphet is a model that combines the power of neural networks with the interpretability of time series models, making it well-suited for time series forecasting tasks.

### Project Structure

The repository is organized as follows:

- `data/`: This directory contains the dataset used for training and testing the models.

- `model/`: An empty directory to contain the trained NeuralProphet model.

- `model_training/`: This directory contains a script for loading data and another for training models.

- `main.py`: This file contains the lines of code used to train the models and display the predictions on a graph.


### Results


Below, the forecasts obtained with the different models.

<div align='center'>

#### Linear Regression
<img src="static/linear_regression_forecasting.png" alt="Linear Regression Forecasting" width="700px" height="350px">
</div>

<br>

<div align='center'>

#### XGBoost
<img src="static/xgboost_forecasting.png" alt="XGBoost Forecasting" width="700px" height="350px">
</div>

<br>

<div align='center'>

#### NeuralProphet
<img src="static/neural_network_forecasting.png" alt="Neural Network Forecasting" width="700px" height="350px">
</div>

<br>

It would seem that the first two models, which are based on several variables (sea ​​level pressure, wind direction and speed, ...), are more precise than the last model, which is only based on the past evolution of the temperature.
