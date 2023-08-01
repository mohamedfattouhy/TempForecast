# Manage environment
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from forecasting_temp import forecasting_temperature


# Create a StandardScaler object
scaler = StandardScaler()

# Read data
path_to_data = os.path.join('data', 'temp_data.csv')
df = pd.read_csv(path_to_data)

# Manage Date
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date


data = df.copy()
data = data.groupby('date', as_index=False).mean()
data.set_index(['date'], inplace=True)

# Apply scaling to DataFrame
data.iloc[:, :-1] = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-1]),
                                 columns=data.columns[:-1],
                                 index=data.index)

# Temperature Forecasting
forecasting_temperature(data=data, method='xgboost')
forecasting_temperature(data=data, method='regression')

data2 = df.copy()
data2 = data2[['date', 'temp_celsius']]
data2 = data2.groupby('date', as_index=False).mean()
data2.columns = ['ds', 'y']

# # Temperature Forecasting
forecasting_temperature(data=data2, method='NeuralProphet')
