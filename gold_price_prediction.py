import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data collection and preprocessing
gold_data = pd.read_csv('gld_price_data.csv')

# Print 5 rows in the data
print(gold_data.head())

# Printing the last 5 rows
print(gold_data.tail())

# To find how many rows and columns are there
print(gold_data.shape)

# To check the info in the given dataset
print(gold_data.info())

# To check whether there is any missing values in the datasets
print(gold_data.isnull().sum())

# Getting the statistics of the data in the csv file
print(gold_data.describe())

# Correlation analysis
correlation = gold_data.select_dtypes(include=['float64', 'int64']).corr()

# Create a heatmap
#plt.figure(figsize=(10,10))
#sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
#plt.show()

# Now I will check the correlation of gold
print(correlation['GLD'])

# Checking the range of values
#sns.histplot(gold_data['GLD'], color='green')
#plt.show()

# Splitting the columns
x = gold_data.drop(['Date','GLD'], axis=1)
y = gold_data['GLD']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train, y_train)

# Predicting values
test_data_prediction = regressor.predict(x_test)

# Print predictions
print(test_data_prediction)

# R squared error value
err_va = metrics.r2_score(y_test, test_data_prediction)
print("The value of R Squared Error is:", err_va)

# Plotting actual vs predicted values
plt.plot(y_test.values, color='orange', label='Actual Value')
plt.plot(test_data_prediction, color='red', label='Predicted Value')
plt.title('Actual Value Vs Predicted Value')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
