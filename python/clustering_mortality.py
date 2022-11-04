'''
Author: Eric Reschke
Cite: https://metricsnavigator.org/mortality-rates/
Last Reviewed: 2022-10-23
License: Open to all
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dataImport = pd.read_csv('1. pollution_data.csv')
df_part1 = pd.DataFrame.copy(dataImport)
df_part2 = pd.DataFrame.copy(dataImport)

colNames = []
for col in df_part1.columns:
    colNames.append(col)

# -------------------------------#
# Part1: clustering on raw data input
    
km = KMeans(n_clusters=3)
part1_predicted = km.fit_predict(df_part1[['Precipitation','JanuaryF','JulyF','Over65','Household','Education','Housing',
'Density','WhiteCollar','LowIncome','HC','NOX','SO2','Humidity','Mortality']])

df_part1['cluster'] = part1_predicted


# -------------------------------#
# Part2: clustering with min/max scaling

# normalizing data
scaler = MinMaxScaler()

# loop for scaling each column
for i in colNames:
    scaler.fit(df_part2[[i]])
    newCol = i+'_scale'
    df_part2[newCol] = scaler.transform(df_part2[[i]])

km = KMeans(n_clusters=3)
part2_predicted = km.fit_predict(df_part2[['Precipitation_scale','JanuaryF_scale','JulyF_scale','Over65_scale',
                                           'Household_scale','Education_scale','Housing_scale','Density_scale',
                                           'WhiteCollar_scale','LowIncome_scale','HC_scale','NOX_scale',
                                           'SO2_scale','Humidity_scale','Mortality_scale']])

df_part2['cluster'] = part2_predicted


# -------------------------------#
# Graphing results, Part 1:

'''
# assigning each cluster to its own dataframe for scatter plot
p1df1 = df_part1[df_part1.cluster==0]
p1df2 = df_part1[df_part1.cluster==1]
p1df3 = df_part1[df_part1.cluster==2]

p1c1 = plt.scatter(p1df1.Precipitation,p1df1['Mortality'],color='green')
p1c2 = plt.scatter(p1df2.Precipitation,p1df2['Mortality'],color='red')
p1c3 = plt.scatter(p1df3.Precipitation,p1df3['Mortality'],color='black')
plt.xlabel('Precipitation')
plt.ylabel('Mortality')
plt.legend([p1c1,p1c2,p1c3],['Cluster 1','Cluster 2','Cluster 3'])

# measuring the success rate of increasing clusters
p1_clusterNumberPlot =[]
p1_clusterIntertia =[]
for i in range(10):
    i+=1
    z = KMeans(n_clusters=i)
    z.fit_predict(df_part1[['Precipitation','JanuaryF','JulyF','Over65','Household','Education','Housing',
                                'Density','WhiteCollar','LowIncome','HC','NOX','SO2','Humidity','Mortality']])
    print(np.round(z.inertia_,0),"cluser of:",i)
    p1_clusterNumberPlot.append(i)
    p1_clusterIntertia.append(np.round(z.inertia_,0))

# simple line graph to show cluster progression
plt.plot(p1_clusterIntertia,linestyle='dotted')

# more detailed plot showing the cluster values
plt.plot(p1_clusterNumberPlot,p1_clusterIntertia,'bo-')
for x,y in zip(p1_clusterNumberPlot,p1_clusterIntertia):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
plt.show()
'''


# -------------------------------#
# Graphing results, Part 2:

'''
# assigning each cluster to its own dataframe for scatter plot
p2df1 = df_part2[df_part2.cluster==0]
p2df2 = df_part2[df_part2.cluster==1]
p2df3 = df_part2[df_part2.cluster==2]

p2c1 = plt.scatter(p2df1.LowIncome_scale,p2df1['Mortality'],color='green')
p2c2 = plt.scatter(p2df2.LowIncome_scale,p2df2['Mortality'],color='red')
p2c3 = plt.scatter(p2df3.LowIncome_scale,p2df3['Mortality'],color='black')
plt.xlabel('LowIncome')
plt.ylabel('Mortality')
plt.legend([p2c1,p2c2,p2c3],['Cluster 1','Cluster 2','Cluster 3'])

# measuring the success rate of increasing clusters
p2_clusterNumberPlot =[]
p2_clusterIntertia =[]
for i in range(10):
    i+=1
    z = KMeans(n_clusters=i)
    z.fit_predict(df_part2[['Precipitation_scale','JanuaryF_scale','JulyF_scale','Over65_scale',
                                           'Household_scale','Education_scale','Housing_scale','Density_scale',
                                           'WhiteCollar_scale','LowIncome_scale','HC_scale','NOX_scale',
                                           'SO2_scale','Humidity_scale','Mortality_scale']])
    print(np.round(z.inertia_,0),"cluser of:",i)
    p2_clusterNumberPlot.append(i)
    p2_clusterIntertia.append(np.round(z.inertia_,0))

# simple line graph to show cluster progression
plt.plot(p2_clusterIntertia,linestyle='dotted')

# more detailed plot showing the cluster values
plt.plot(p2_clusterNumberPlot,p2_clusterIntertia,'bo-')
for x,y in zip(p2_clusterNumberPlot,p2_clusterIntertia):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
plt.show()
'''


## end of script

