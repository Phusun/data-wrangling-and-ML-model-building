# This file makes PREDICTIONS for test data that is randomly sampled from the given data set

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


# function for predicting

def predictor(mlModel='RandomForest', forecastYear=1, predX = None):
    # Extract X,y data
    dataXY = data.drop(columns=['Year','Metro name'])
    # drop rows with response value NaN for the given forecast year
    dataXY.dropna(subset=[dataXY.columns[-(4-forecastYear)]], inplace=True)
    # data with no NaN in y
    X = dataXY.iloc[:,:-3]
    y = dataXY.iloc[:, [-(4-forecastYear)]]

    # select ML model with best parameters for prediction
    if mlModel == 'RandomForest':
        reg = RandomForestRegressor(**rfBestParams[forecastYear-1])
    elif mlModel == 'xgboost':
        reg = xgboost.XGBRegressor(objective ='reg:squarederror', random_state=random_state)
    else:
        reg = GradientBoostingRegressor(**gbBestParams[forecastYear-1])

    if predX is None:
        # if no prediction data is given, create a train/test split for prediction and record the indices
        indices = np.arange(len(X))
        trainX, testX, trainY, testY, idx1, idx2 = train_test_split(X, y, indices, train_size=0.9, random_state=random_state)
        # model training on training split
        reg.fit(trainX, trainY.values.ravel())

        # prediction on test split
        predY = reg.predict(testX)

        # compute and print prediction performance (mean absolute error)
        predMAE = mean_absolute_error(testY, predY)
        print('{}-year forecasting MAE with {}: '.format(str(forecastYear), mlModel), predMAE)

        # write the prediction results in a csv file
        df_result = data.loc[list(idx2), ['Metro name', dataXY.columns[-(4-forecastYear)]]]
        df_result['prediction'] = predY
        df_result['error (%)'] = df_result.apply(lambda row: round((row['prediction']-row[dataXY.columns[-(4-forecastYear)]])/row[dataXY.columns[-(4-forecastYear)]]*100, 2), axis=1)
        print(df_result)
        df_result.to_csv('Prediction_results.csv', index=False)

        # plot the actual and predicted test results
        plt.xticks(list(range(len(testX))), list(df_result['Metro name']), rotation=-90)
        plt.plot(list(range(len(testX))), testY, 'b.--', label='Test data')
        plt.plot(list(range(len(testX))), predY, 'r+-', label='Prediction')
        plt.title('Sale to List Ratio {}-year Prediction'.format(str(forecastYear)), fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

    else:
        # model training on entire data
        reg.fit(X, y.values.ravel())

        # prediction on the given data
        predY = reg.predict(predX)

    # feature importance in descending order
    print('FEATURE IMPORTANCE')
    indices = np.argsort(reg.feature_importances_)[::-1]
    for idx in list(indices):
        print('{}: {:.1f}%'.format(X.columns[idx], reg.feature_importances_[idx]*100))

    return predY

# With missing predictor data, only use xgboost as the mlModel
predY = predictor(mlModel='xgboost', forecastYear=3, predX = None)



# use of GridSearchCV to select the best parameters for each model

#*************************** RANDOM FOREST*******************************
# parameters = {'n_estimators': list(range(50,150,10)),'max_depth': list(range(5,35,5))}

# rf = RandomForestRegressor(random_state=random_state)

# regRF = GridSearchCV(rf, parameters, cv=5)

# regRF.fit(trainX, trainY.values.ravel())

# reg = regRF.best_estimator_

#**************************GRADIENT BOOSTING*****************************

# parameters = {'n_estimators': list(range(100,510,10)),'max_depth': list(range(2,7))}

# gbr = GradientBoostingRegressor(random_state=random_state)

# regGBR = GridSearchCV(gbr, parameters, cv=5)

# regGBR.fit(trainX, trainY.values.ravel())

# reg = regGBR.best_estimator_
