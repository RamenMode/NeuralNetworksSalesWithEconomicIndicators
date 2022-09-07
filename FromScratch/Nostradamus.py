def nostradamus(dataframe, baselinedataframe, horizon, numepochs, batchsize, regindex):
    """ OVERVIEW OF THE INPUT VARIABLES OF THE FUNCTION nostradamus:

    dataframe: A dataframe which contains dates as the index and sales data

    baselinedataframe: A dataframe which contains (AND MUST CONTAIN) the same dates as dataframe, expanded dates in the form dayof the
    week, day of the month, etc, and labels for the various holidays.

    horizon: A dataframe which contains future dates

    firstlayer: The number of nodes in the first hidden layer of the neural net

    secondlayer: The number of nodes in the second hidden layer of the neural net

    lastlayer: The number of nodes in the output layer of the neural nets. It always equals the number of data
    columns.

    numepochs: The number of epochs the neural net will be allowed to run.

    batchsize: The epochs are run in batches. This parameter specifies how many dates we are going to process in a
    single batch.

    regindex: The regularizer keeps the gradient from becoming too large or two small. The regindex specifies the
    intensity of this mechanism.


    OUTPUT VALUES OF nostradamus:

    The function spits out the dataframe newpredictionspdf, containing sales prediction for the period specified in
    the dataframe horizons. It also spits out the mean square error.


    """
    # dates=dataframe.index

    Y = dataframe
    X = baselinedataframe

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    X_num_columns = len(X.columns)

    model = Sequential()
    model.add(Dense(20,
                    activation='relu',
                    input_dim=X_num_columns,

                    kernel_regularizer=keras.regularizers.l2(regindex)
                    ))

    model.add(Dense(10,
                    activation='relu',

                    kernel_regularizer=keras.regularizers.l2(regindex)
                    ))
    model.add(Dropout(0.2))

    # model.add(Dense(200,
    # activation='relu',

    # kernel_regularizer=keras.regularizers.l2(regindex)
    # ))
    # model.add(Dropout(0.2))

    # model.add(Dense(100,
    # activation='relu',

    # kernel_regularizer=keras.regularizers.l2(regindex)
    # ))

    # model.add(Dropout(0.2))

    # model.add(Dense(7,
    # activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(1,
                    activation='linear',

                    kernel_regularizer=keras.regularizers.l2(regindex)
                    ))

    # optimizer = keras.optimizers.Adam(lr=0.1)

    # optimizer = keras.optimizers.Adam(learning_rate=.001)
    model.compile(optimizer='Adam', loss='mse')

    # Fit model to training data

    history = model.fit(X_train, y_train, epochs=numepochs, batch_size=batchsize, verbose=2)

    loss = model.evaluate(X_test, y_test)

    predictions = model.predict(X)

    predictionsdf = pd.DataFrame(data=predictions, index=X.index)

    predictionsdf.columns = Y.columns

    # dates_series =  pd.Series(dates)

    df_newDates = horizon

    # Predict upcoming sales using trained model and imported upcoming dates

    Predicted_sales = model.predict(df_newDates)

    newpredictionsdf = pd.DataFrame(data=Predicted_sales)

    newpredictionsdf.columns = Y.columns

    return [newpredictionsdf, history, np.sqrt(loss)]