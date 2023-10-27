'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/mortality-rates-deep-dive/
Last Reviewed: 2023-10-27
License: Open to all
'''

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

NaN = np.nan

# iterating the columns
def colReveal(x):
    for col in x.columns:
        print(col)

# z-score function
def zScores(df,col):
    if (len(df)<=1):
        print('df must have more than one row')
    else:
        samplePop = len(df)-1
        avg = np.mean(df[col])
        varEach = (df[col]-avg)**2
        var = sum(varEach)/(samplePop)
        sd = np.sqrt(var)
        df[col+'zScore'] = (df[col]-avg)/sd
        return(df)

dataImport = pd.read_csv('1. pollution_data.csv')
df_training = pd.DataFrame.copy(dataImport)

# assign z-scores to each metric along with cohorts
for i in df_training:
    zScores(df_training,str(i))
    df_training[str(i)+'_cohort'] = np.floor(abs(df_training[str(i)+'zScore'])+1)

df_training = df_training[['Precipitation_cohort','JanuaryF_cohort','JulyF_cohort','Over65_cohort','Household_cohort',
                           'Education_cohort','Housing_cohort','Density_cohort','WhiteCollar_cohort','LowIncome_cohort',
                           'HC_cohort','NOX_cohort','SO2_cohort','Humidity_cohort','Mortality_cohort']]

# copying dr_training in its current state for different networks
df_part1 = df_training.copy()

# neural network
# layer size = sample / (scaler*(inputs+outputs))

sample = len(dataImport)
inputs = len(dataImport.columns)-1
outputs = 1
scaler = 1
hidden_num = round(sample / (scaler*(inputs+outputs)),0)

randState=23

# -------------------------------#
# Part1: using integer on raw data

# use hidden_num to determine the amount of hidden layers
classifier = MLPClassifier(hidden_layer_sizes=(300,200,150,50),
                           max_iter=500,activation = 'relu',solver='adam',
                           random_state=randState)

X = dataImport.iloc[:,0:-1].values.astype('int')
Y = dataImport.iloc[:,-1].values.astype('int')

classifier.fit(X,Y)

y_pred = classifier.predict(X)
y_export = pd.DataFrame(y_pred)
y_export.columns=['y_pred']

df_part1['final_Y'] = y_export['y_pred']
df_part1['Mortality'] = round(dataImport['Mortality'],0)

# export the initial run to csv
df_part1.to_csv('mortality_rate_predictions_part1.csv',index=False)


# -------------------------------#
# Part2: using cohorts

df_part2 = df_training.copy()

classifier = MLPClassifier(hidden_layer_sizes=(300,200,150,50),
                           max_iter=500,activation = 'relu',solver='adam',
                           random_state=randState)

X = df_part2.iloc[:,0:-1].values
Y = df_part2.iloc[:,-1].values

classifier.fit(X,Y)

y_pred = classifier.predict(X)
y_export = pd.DataFrame(y_pred)
y_export.columns=['y_pred']

df_part2['Predicted_cohort'] = y_export['y_pred']
df_part2['Mortality'] = round(dataImport['Mortality'],0)

# export the initial run to csv
df_part2.to_csv('mortality_rate_predictions_part2.csv',index=False)

# accuracy check on part2 run
df_part2['Accuracy'] = np.where(df_part2['Mortality_cohort']==df_part2['Predicted_cohort'],1,0)
mlp_training_accuracy = round(np.sum(df_part2['Accuracy'])/len(df_part2),4)
print('\nInitial Run Accuracy:',mlp_training_accuracy,'\n')

# confusion matrix
cTrainingMatrix = confusion_matrix(df_part2['Mortality_cohort'],df_part2['Predicted_cohort'])
cTrainingMatrix = pd.DataFrame(cTrainingMatrix)
RunError = round(cTrainingMatrix[0][1]/cTrainingMatrix[1][1],2)
    
# training confusion matrix
print('--- Confusion Matrix ---','\n')
print('Left Side Actuals; Headers Predictions\n\n',cTrainingMatrix,'\n')

# counts of accurate and missed cohort predictions
missedPred = cTrainingMatrix[0][1]+cTrainingMatrix[0][2]+cTrainingMatrix[1][0]+cTrainingMatrix[1][2]+cTrainingMatrix[2][0]+cTrainingMatrix[2][1]
print('Count Accurate:',len(df_part2)-missedPred,'\nCount Missed:',missedPred)


## end of script

