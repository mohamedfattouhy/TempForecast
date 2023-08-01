"""This file contains the code to train the various forecasting models"""

# Manage Environment
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Set 'bmh' style for plot
plt.style.use('bmh')


def forecasting_temperature(data: pd.DataFrame, method: str):
    """Make temperature forecasts with different models"""

    train_size = int(0.8*data.shape[0])
    test_size = data.shape[0] - train_size

    train = data.iloc[:train_size, :]
    test = data.iloc[-test_size:, :]

    # Linear Regression Method
    if method == 'regression':

        reg = LinearRegression()
        reg.fit(train.iloc[:, :-1], train['temp_celsius'])
        predictions = reg.predict(test.iloc[:, :-1])

        data_pred = pd.DataFrame(predictions, index=test.index)
        data_pred.columns = ["yhat"]

        plt.plot(train.index,
                 train.iloc[:, train.columns.get_loc('temp_celsius')],
                 color='#E74C3C')

        plt.plot(test.index,
                 test.iloc[:, test.columns.get_loc('temp_celsius')],
                 color='#2980B9', label='real')

        plt.plot(data_pred.index, data_pred['yhat'],
                 color='#7D3C98', label='prediction')

        plt.xlabel('Date')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Temperature evolution')
        plt.legend()
        plt.axvline(x=pd.to_datetime('2023-01-18'), color='black', ls='--')

        plt.show()

    # xgboost method
    if method == 'xgboost':

        reg = xgb.XGBRegressor(n_estimators=100)

        reg.fit(train.iloc[:, :-1], train['temp_celsius'])
        predictions = reg.predict(test.iloc[:, :-1])

        data_pred = pd.DataFrame(predictions, index=test.index)
        data_pred.columns = ["yhat"]

        plt.plot(train.index,
                 train.iloc[:, train.columns.get_loc('temp_celsius')],
                 color='#E74C3C')

        plt.plot(test.index,
                 test.iloc[:, test.columns.get_loc('temp_celsius')],
                 color='#2980B9', label='real')

        plt.plot(data_pred.index, data_pred['yhat'],
                 color='#7D3C98', label='prediction')

        plt.xlabel('Date')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Temperature evolution')
        plt.legend()
        plt.axvline(x=pd.to_datetime('2023-01-18'), color='black', ls='--')

        plt.show()

    # Neural Network Method
    if method == 'NeuralProphet':

        neural_model = NeuralProphet()
        neural_model.fit(df=train, freq='D', epochs=100)

        save_model = os.path.join('model', 'neural_network_model.pkl')

        with open(save_model, "wb") as f:
            pickle.dump(neural_model, f)

        trained_model_path = os.path.join('model', 'neural_network_model.pkl')

        with open(trained_model_path, "rb") as f:
            trained_model = pickle.load(f)

        future_date = trained_model.\
            make_future_dataframe(df=train,
                                  periods=test_size)
        trained_model.restore_trainer()
        forecast = trained_model.predict(future_date)

        train.set_index(['ds'], inplace=True)
        test.set_index(['ds'], inplace=True)
        forecast.set_index(['ds'], inplace=True)

        plt.plot(train.index,
                 train.iloc[:, train.columns.get_loc('y')],
                 color='#E74C3C')

        plt.plot(test.index,
                 test.iloc[:, test.columns.get_loc('y')],
                 color='#2980B9', label='real')

        plt.plot(forecast.index, forecast['yhat1'],
                 color='#7D3C98', label='prediction')

        plt.xlabel('Date')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Temperature evolution')
        plt.legend()
        plt.axvline(x=pd.to_datetime('2023-01-18'), color='black', ls='--')

        plt.show()
