#! pip install prophet # Only use this line if prophet is not already installed

import pandas as pd
from prophet import Prophet

#Use this this to predict
# pred_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")


training_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
training_data.Timestamp = pd.to_datetime(training_data.Timestamp,infer_datetime_format=True)


# Keep only the dates and the y value
data = training_data[['Timestamp','trips']]
data = pd.DataFrame(data.values, columns = ['ds','y'])

# Initialize Prophet instance and fit to data

model = Prophet(changepoint_prior_scale=0.5)
modelFit = model.fit(data)


# Create timeline for 1 year in future, 
#   then generate predictions based on that timeline

future = modelFit.make_future_dataframe(periods=31*24,freq='H')
forecast = modelFit.predict(future)
forecast


pred = forecast[forecast['ds'] > modelFit.history['ds'].max()][['ds','trend']]
pred['trend'] = pred['trend'].astype(int)
pred


# Create plots of forecast and truth, 
#   as well as component breakdowns of the trends
plt = modelFit.plot(forecast)
plt.savefig("prophet.png")
comp = modelFit.plot_components(forecast)