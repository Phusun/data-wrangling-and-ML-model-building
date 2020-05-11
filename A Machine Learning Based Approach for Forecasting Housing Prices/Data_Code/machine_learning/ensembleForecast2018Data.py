# This file makes FORECAST for test data that is 1, 2 or 3 years in future for a given year

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

random_state = 42

# read the complete dataset
data = pd.read_csv('../databin/ml_dataset.csv')

data.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)

# best parameters of Random Forest and Gradient Boosting Regressor models - obtained from GridSearchCV
rfBestParams = [{'max_depth': 10, 'n_estimators': 60, 'random_state': random_state}, {'max_depth': 20, 'n_estimators': 110, 'random_state': random_state}, {'max_depth': 10, 'n_estimators': 60, 'random_state': random_state}]

gbBestParams = [{'max_depth': 3, 'n_estimators': 230, 'random_state': random_state}, {'max_depth': 4, 'n_estimators': 300, 'random_state': random_state},{'max_depth': 4,'n_estimators': 270, 'random_state': random_state}]

# function for forecasting

def forecaster(mlModel='RandomForest', year=2018):
    # move the Metro name column to the front
    dataXY = data.copy()
    cols = list(dataXY.columns)
    cols = [cols[-1]] + cols[:-1]
    dataXY = dataXY[cols]
    dataXY.head()

    df_forecast = None

    for forecastYear in [1,2,3]:
        # test data is always the same for the given year
        testXY = dataXY[dataXY['Year']==year]
        testX = testXY.iloc[:,:-3]
        testX.drop(columns=['Year','Metro name'], inplace=True)

        # drop rows with response value NaN for the given forecast year
        dataXYi = dataXY.dropna(subset=[dataXY.columns[-(4-forecastYear)]])
        # data with no NaN in y
        X = dataXYi.iloc[:,:-3]
        y = dataXYi.iloc[:, [-(4-forecastYear)]]

        # select ML model with best parameters for prediction
        if mlModel == 'RandomForest':
            reg = RandomForestRegressor(**rfBestParams[forecastYear-1])
        elif mlModel == 'xgboost':
            reg = xgboost.XGBRegressor(objective ='reg:squarederror')
        else:
            reg = GradientBoostingRegressor(**gbBestParams[forecastYear-1])

        # select the training and test data
        trainX = X[X['Year']!=year]
        trainY = y[X['Year']!=year]


        # after filtering data by year, we can drop metro name and year columns as before
        trainX.drop(columns=['Year','Metro name'], inplace=True)

        # model training on training split
        reg.fit(trainX, trainY.values.ravel())

        # prediction on test split
        predY = reg.predict(testX)

        if df_forecast is None:
            df_forecast = dataXY.loc[dataXY['Year']==year, ['Metro name']]
            df_forecast[year+forecastYear] = predY
        else:
            df_forecast[year+forecastYear] = predY

        # write forecasting results in a csv
        df_forecast.to_csv('Forecasting based on {} data.csv'.format(year), index=False)

# With missing predictor data, only use xgboost as the mlModel
forecaster(mlModel='xgboost', year=2018)
