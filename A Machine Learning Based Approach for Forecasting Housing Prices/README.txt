## Data
A variety of data sources were used to pull together all the features utilized in the machine learning algorithm. 
Housing data was pulled from:
	https://www.redfin.com/blog/data-center/
	https://www.zillow.com/research/data/
Stock market data was pulled from:
	https://finance.yahoo.com/most-active
Macroeconomic data was pulled from:
	https://fred.stlouisfed.org/
	https://www.governing.com/gov-data/other/city-construction-building-permits-data-totals.html
	https://www.bls.gov/bls/news-release/metro.htm
	https://www.fhfa.gov/DataTools/Downloads/Pages/House-Price-Index-Datasets.aspx
	http://www.freddiemac.com/research/indices/house-price-index.page
	https://www.bls.gov/eag/
	https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas
	https://www.census.gov/housing/hvs/data/ann15ind.html.

## Data Manipulation
buildDataset.py was designed to build a clean CSV of X and Y values for the machine learning algorithm.
It lives in Data_Code/databin/ and reads all CSV files under "Data_Code/databin/Clean Macro data/"
It pulls their 2010-2018 data and assembles all that data for each feature into one column.
It then combines all those columns.
To add the Y columns, the script pulls the sale-to-list ratio data again, but shifts it by removing the appropriate years and padding the bottom with "NaN" values.
For example, for the 2 year prediction, 2010 and 2011 data was removed, and 2 years of "NaN" values were added.
The Y columns were then combined with the feature data to get the final machine learning dataset.
For convenience, the data was sorted by year, and each row was tagged with a metropolitan area.
While this metadata was not used for machine learning, it made it very easy to avoid peeking into the future when training out forecasting model.
It also helped to run the individual metro area predictions for 2019, 2020, and 2021.
The beauty of buildDataset.py is that, to add a new feature to the model, the CSV just needs to be dropped into "Data_Code/databin/Clean Macro data/" and the script re-run.
It outputs the new dataset as ml_dataset.csv under "Data_Code/databin"

## Model training for prediction and forecast
The forecast values for sale to list ratio are generated from a trained machine learning (ML) model that can handle mising or "NaN" values in the training data stored in ml_dataset.csv as well as can deal with different scales of the features/predictors without the need to standardize them. Model also outputs a list of importance of different features for the task of prediction and forecasting.
There are three different ML model scripts and they can be found in "Data_Code/machine_learning/". ensemblePrediction.py is used for prediction (baseline), ensembleForecast.py is used for forecast (innovation) and both plot the results, print a table of results and a list of feature importance as well as write the results in a CSV. Main function in all three accepts one of the three ML algorithms, 1) Random Forest 2) Gradient Boosting 3) xgboost as an argument, however only xgboost is capable of handling the "NaN" values, therefore it has been set as the deafult. If predictor data doesn't contain any "NaN" or missing values, any of three algorithms can be selected. Other arguments to the function are years ahead for prediction/forecast and any explicit test dataset that is not part of the original data.
GridSeachCV in scikit-learn library is used to come up with the best set of parameters based on cross-validation and these parameters are stored as dictionaries in the script. For user-specified 1, 2 or 3 years prediction/forecast, a different model, regardless of the algorithm chosen, is trained and its best parameters are loaded from the appropriate dictionary. This ensures the high level of performance in all scenarios.
Very important: if you want to use data with missing predictor/feature values, only use xgboost as your ML model choice. Unavailable Y data for certain years is not an issue and all scripts can handle it.
A more detailed description of each of the three ML scripts follow below:

	a) ensemblePrediction.py: makes predictions for a random test split of the given data. These predictions are not true forecasts since there are randomly selected data points. This is our benchmark for this project since most other work on this topic are about predicting with ML.

	b) ensembleForecast.py: makes 1, 2 and 3 years forecasts based on the data for the prior year(s). For ex., a 1, 2 and 3 year forecast based on 2018 will be sale to list ratio for 2019, 2020 and 2021.

	c) ensembleForecast2018Data.py: makes 1, 2 and 3 years forecast for a given year. Although the given year is set as 2018, it can be changed by the user inside the script. Uses all the prior data to train the models.
