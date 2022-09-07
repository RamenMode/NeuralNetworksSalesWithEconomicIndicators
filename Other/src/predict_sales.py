import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
import sys
import time
from multiprocessing import Pool
from copy import deepcopy
import csv

def create_model(numNodes, includeIndicator=True, activation='relu',
                 optimizer='adam', loss='mean_squared_error'):
    """
    Create Dense neural network model for evaluating each indicator's explanatory strength
    
    inputs
    --
    numNodes (int) : baseline number of nodes used for constructing the neural network layers
    includeIndicator (bool) : Used to mark whether a model is fitted with an indicator feature
    activation, optimizer, loss (string) : Tensorflow neural network parameters
    
    outputs
    --
    model (tf.keras.Model()) : neural network model
    """
    
    
    model = Sequential()
    model.add(Dense(numNodes*5/3, activation = activation, input_dim=2+int(includeIndicator))) # increase input_dim
#     model.add(Dense(numNodes*2/3, activation = activation))
    model.add(Dropout(0.2))
    model.add(Dense(numNodes*1/3, activation = activation))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def run(SalesColName, mrts_data, TestYears, NumNodes):
    file = "Results/resultsMRTS.csv"
    fields = ["SalesColumn", "Indicator", "TrainYearStart", "TestYear", "TrainStDev", "TestRMSE", "ExplVarRatio"]
    with open(file, 'a') as f:
        write = csv.writer(f)
        write.writerow(fields)

        print(SalesColName)
        for i, indicator_file in enumerate(glob.glob("Data/Economic indicators/*")):
            ind_name = indicator_file.split("/")[-1]

            indicator = pd.read_csv(indicator_file)
            indicator['Year'] = indicator['DateTime'].apply(lambda x: int(x.split("-")[0]))
            indicator['Month'] = indicator['DateTime'].apply(lambda x: int(x.split("-")[1]))
            indicator = indicator[["Year", "Month", "Value"]].rename(columns={"Value":"Indicator"})
            new_sales_col = []
            for idx, row in indicator.iterrows():
                try:
                    sales = mrts_data[((mrts_data['Year'] == row.Year) &
                                    (mrts_data['Month'] == row.Month))][SalesColName].tolist()[0]
                    new_sales_col.append(sales)
                except IndexError:
                    new_sales_col.append(sales)
                    continue
            indicator["Sales"] = new_sales_col
            data = indicator.copy()
            
            for testYear in TestYears:
                trainSet = data[data["Year"]<testYear]
                X_train = trainSet[["Month", "Year", "Indicator"]].to_numpy()
                y_train = trainSet["Sales"].to_numpy()
                
                testSet = data[data["Year"]==testYear]
                X_test = testSet[["Month", "Year", "Indicator"]].to_numpy()
                y_test = testSet["Sales"].to_numpy()
                
                model = create_model(NumNodes)
                model.fit(X_train, y_train, epochs=500, batch_size=128, verbose=0) #change batch size to a variable
        
                # Test results from model using training data
                y_preds = model.predict(X_test)
                testRMSE = mean_squared_error(y_test, y_preds, squared=False)
                trainStDev = trainSet["Sales"].std()
                evr = (trainStDev - testRMSE) / trainStDev
                
                del model
                
                # resultsMRTS["Sales Column"].append(SalesColName)
                # resultsMRTS["Indicator"].append(ind_name)
                # resultsMRTS["Train Year Start"].append(trainSet["Year"].min())
                # resultsMRTS["Pred Year"].append(int(testYear))
                # resultsMRTS["Train Std"].append(round(trainStDev, 3))
                # resultsMRTS["Test RMSE"].append(round(testRMSE, 3))
                # resultsMRTS["Explained Variance Ratio"].append(round(evr, 3))
                to_write = [SalesColName.replace(" ", "_").replace(",",""), ind_name, trainSet["Year"].min(),
                            testYear, round(trainStDev,3), round(testRMSE, 3), round(evr,3)]
                write.writerow(to_write)
    f.close()
    return

if __name__=="__main__":

    months = {
        "JAN":1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6,
        "JUL":7, "AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12
    }

    resultsMRTS = {
                "Sales Column": [],
                "Indicator" : [],
                "Train Year Start": [],
                "Pred Year": [],
                "Train Std": [], 
                "Test RMSE": [],
                "Explained Variance Ratio": []
    }

    mrts_data = pd.read_csv("Data/mrtssales_92-present.csv")
    mrts_data['Date'] = mrts_data.iloc[:,0].apply(lambda x: x.replace(".", ""))
    mrts_data["Month"] = mrts_data["Date"].apply(lambda x: x[:3].upper()).tolist()
    mrts_data["Year"] = mrts_data["Date"].apply(lambda x: x[4:].upper()).tolist()
    mrts_data = mrts_data.drop([mrts_data.columns[0],"Date"], axis=1)
    mrts_data = mrts_data[:-1] # delete present month
    mrts_data["Year"] = mrts_data["Year"].astype(int)
    mrts_data["Month"] = mrts_data["Month"].apply(lambda x: months[x])

    TestYears = [2018, 2019, 2020]
    NumNodes = 153 # approx 2/3 of the number of your rows. Make this divisible by a three

    for col in mrts_data.columns:
        run(col, mrts_data, TestYears, NumNodes)
