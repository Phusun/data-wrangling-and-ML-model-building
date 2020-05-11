# This file makes FORECAST for test data that is 1, 2 or 3 years in future

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

# year(s) to exclude for true forecasting
yearExcld = [2018, 2017, 2016]


# function for forecasting

def forecaster(mlModel='RandomForest', forecastYear=1, predX = None):
    # move the Metro name column to the front
    dataXY = data.copy()
    cols = list(dataXY.columns)
    cols = [cols[-1]] + cols[:-1]
    dataXY = dataXY[cols]
    dataXY.head()

    # drop rows with response value NaN for the given forecast year
    dataXY.dropna(subset=[dataXY.columns[-(4-forecastYear)]], inplace=True)
    # data with no NaN in y
    X = dataXY.iloc[:,:-3]
    y = dataXY.iloc[:, [-(4-forecastYear)]]

    # select ML model with best parameters for prediction
    if mlModel == 'RandomForest':
        reg = RandomForestRegressor(**rfBestParams[forecastYear-1])
    elif mlModel == 'xgboost':
        reg = xgboost.XGBRegressor(objective ='reg:squarederror')
    else:
        reg = GradientBoostingRegressor(**gbBestParams[forecastYear-1])

    if predX is None:
        trainX = X[X['Year']!=yearExcld[forecastYear-1]]
        testX = X[X['Year']==yearExcld[forecastYear-1]]
        trainY = y[X['Year']!=yearExcld[forecastYear-1]]
        testY = y[X['Year']==yearExcld[forecastYear-1]]

        # after filtering data by year, we can drop metro name and year columns as before
        trainX.drop(columns=['Year','Metro name'], inplace=True)
        testX.drop(columns=['Year','Metro name'], inplace=True)
        # model training on training split
        reg.fit(trainX, trainY.values.ravel())

        # prediction on test split
        predY = reg.predict(testX)

        # compute and print prediction performance (mean absolute error)
        predMAE = mean_absolute_error(testY, predY)
        print('{}-year forecasting MAE with {}: '.format(str(forecastYear), mlModel), predMAE)

        # write the forecasting results in a csv file
        df_result = dataXY.loc[dataXY['Year']==yearExcld[forecastYear-1], ['Metro name', dataXY.columns[-(4-forecastYear)]]]
        df_result['forecast'] = predY
        df_result['error (%)'] = df_result.apply(lambda row: round((row['forecast']-row[dataXY.columns[-(4-forecastYear)]])/row[dataXY.columns[-(4-forecastYear)]]*100, 2), axis=1)
        print(df_result)
        print('Average Percent error: ', df_result['error (%)'].mean())
        df_result.to_csv('Forecasting_results.csv', index=False)

        # plot the actual and forecast test results
        plt.xticks(list(range(len(testX))), list(df_result['Metro name']), rotation=-90)
        plt.plot(list(range(len(testX))), testY, 'b.--', label='Test data')
        plt.plot(list(range(len(testX))), predY, 'r+-', label='Forecast')
        plt.title('Sale to List Ratio {}-year Forecast'.format(str(forecastYear)), fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.savefig('results.png')

    else:
        # model training on entire data
        reg.fit(X, y.values.ravel())

        # prediction on the given data
        predY = reg.predict(predX)

    # feature importance in descending order
    print('FEATURE IMPORTANCE')
    indices = np.argsort(reg.feature_importances_)[::-1]
    for idx in list(indices):
        print('{}: {:.1f}%'.format(trainX.columns[idx], reg.feature_importances_[idx]*100))

    return predY

# With missing predictor data, only use xgboost as the mlModel
predY = forecaster(mlModel='xgboost', forecastYear=3, predX = None)
