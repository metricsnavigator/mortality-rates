'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/mortality-rates-deep-dive/
Last Reviewed: 2022-10-19
License: Open to all
'''

import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statsmodels.api as sm

dataImport = pd.read_csv('1. pollution_data.csv')
mortality_df = pd.DataFrame.copy(dataImport)

# convert data to arrays to set up the regression model
X = mortality_df.iloc[:,0:-1].values
Y = mortality_df.iloc[:,-1].values

lin = sklearn.linear_model.LinearRegression()
lin.fit(X,Y)

prediction = lin.predict(X)
FinalPrediction = []
for i in prediction:
    x = round(Decimal(i),0)
    FinalPrediction.append(x)

y_export = pd.DataFrame(FinalPrediction)
y_export.columns=['Prediction']
mortality_df['Prediction'] = y_export['Prediction']

# -------------------------------------------------- #

# regression on mortality rates
model = sm.OLS.from_formula('Mortality ~ Precipitation+JanuaryF+JulyF+Over65+Household+Education+Housing+Density+WhiteCollar+LowIncome+HC+NOX+SO2+Humidity',data=mortality_df)
result = model.fit()
print(result.summary())


## end of script

