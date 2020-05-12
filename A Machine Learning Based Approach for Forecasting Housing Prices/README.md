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
buildDataset.py was designed to build a clean CSV of X and Y values from the raw data.
It lives in Data_Code/databin/ and reads all CSV files under "Data_Code/databin/Clean Macro data/"
It pulls their 2010-2018 data and assembles all that data for each feature into one column.
It then combines all those columns.
To add the Y columns, the script pulls the sale-to-list ratio data again, but shifts it by removing the appropriate years and padding the bottom with "NaN" values.  
To add a new feature to the model, the CSV just needs to be dropped into "Data_Code/databin/Clean Macro data/" and the script re-run.
It outputs the new dataset as ml_dataset.csv under "Data_Code/databin".

## Model training for prediction and forecast
Important: if you want to use data with missing predictor/feature values, only use xgboost as your ML model choice. Unavailable Y data for certain years is not an issue and all scripts can handle it.
A more detailed description of each of the three ML scripts follow below:

1. ensemblePrediction.py: makes predictions for a random test split of the given data. These predictions are not true forecasts since there are randomly selected data points. This is our benchmark for this project since most other work on this topic are about predicting with ML.

2. ensembleForecast.py: makes 1, 2 and 3 years forecasts based on the data for the prior year(s). For ex., a 1, 2 and 3 year forecast based on 2018 will be sale to list ratio for 2019, 2020 and 2021.

3. ensembleForecast2018Data.py: makes 1, 2 and 3 years forecast for a given year. Although the given year is set as 2018, it can be changed by the user inside the script. Uses all the prior data to train the models.
