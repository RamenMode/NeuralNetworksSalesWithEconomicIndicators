import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import sys

months = {"JAN":1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6, 
          "JUL":7, "AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12}

data = pd.read_pickle('MarylandVehicleSales2002-2021')
dataNew = pd.DataFrame(data)
dataNew = dataNew.drop(columns = ['Total Sales New', 'Total Sales Used'])





# dataNew["Month"] = dataNew["Month "].apply(lambda x: months[x])
# X = dataNew[["Month", "Year ", "New"]]
# Y = dataNew["New"].reset_index()

# ind = pd.read_csv("historical_country_United_States_indicator_Capacity_Utilization.csv")
# ind["Date"] = pd.to_datetime(ind["DateTime"]).dt.date
# ind = ind[ind["Date"]>=pd.to_datetime("2002-01-01")]
# ind = ind[ind["Date"]<=pd.to_datetime("2021-05-01")]

# X = X.join(ind["Value"].reset_index()).drop(columns=["index"])
# print(X.head())





column = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    1, 2, 3, 4]
dataNew['Month'] = column

for i in range(0, 232):
    dataNew.iloc[i, 2] = dataNew.iloc[i, 2] + dataNew.iloc[i, 3]
dataNew = dataNew.drop(columns = ['Used', 'Month '])
dataNew = dataNew.rename(columns={'New': 'Total Sales'})

dataNew = dataNew[['Total Sales', 'Year ', 'Month']]
ind = pd.read_csv("historical_country_United_States_indicator_Capacity_Utilization.csv")
ind["Date"] = pd.to_datetime(ind["DateTime"]).dt.date
ind = ind[ind["Date"]>=pd.to_datetime("2002-01-01")]
ind = ind[ind["Date"]<=pd.to_datetime("2021-05-01")]
dataNew = dataNew.join(ind["Value"].reset_index()).drop(columns=["index"])
#What are we testing with?
testData = dataNew.loc[dataNew['Year '] == 2019]
testData = testData.set_index([pd.Index(list(range(12)))])
anotherData = dataNew.set_index('Year ')
trainData = anotherData.drop(2019, axis = 0)
trainData = trainData.reset_index()
trainData = trainData[['Total Sales', 'Year ', 'Month', 'Value']]
dates = trainData.index


#print("dates: ", dates)



# Now we isolate training and test
X = trainData.iloc[:, 0:4]
Y = trainData.iloc[:, 0]
Xi_Test = testData.iloc[:, 0:3]
Yi_Test = testData.iloc[:, 0]



# Model Function
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2, shuffle = True)
model = Sequential()
model.add(Dense(450, activation = 'relu', input_dim=4))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(XTrain, YTrain, epochs=100, batch_size=128)

#results from model
loss = model.evaluate(XTest, YTest)
print('sqrt loss', np.sqrt(loss))
print('standard deviation', trainData['Total Sales'].std())

predictions = model.predict(X)

predictions_list = map(lambda x: x[0], predictions)
predictions_series = pd.Series(predictions_list)
dates_series = pd.Series(dates)

Predicted_sales = model.predict(Xi_Test)
new_dates_series=pd.Series(Xi_Test.index)
new_predictions_list = map(lambda x: x[0], Predicted_sales)
new_predictions_series = pd.Series(new_predictions_list,index=new_dates_series)

#export to csv
new_predictions_series.to_csv("predicted_saless.csv",header=False)

# Author: Maryland Gov
# https://catalog.data.gov/dataset?publisher=opendata.maryland.gov&organization=state-of-maryland
# Used and New Car Data sales monthly
# https://catalog.data.gov/dataset/mva-vehicle-sales-counts-by-month-for-calendar-year-2002-2020-up-to-october
