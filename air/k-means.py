import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
airline_data = pd.read_csv("../data/air_data.csv",encoding="gb18030") 
# print(airline_data.shape)
#airline_data.info()
#airline_data.describe()
#airline_data.head(10)

exp1 = airline_data["SUM_YR_1"].notnull()
exp2 = airline_data["SUM_YR_2"].notnull()
exp = exp1 & exp2
airline_notnull = airline_data.loc[exp,:]
# print(,airline_notnull.shape)

index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM']> 0) & (airline_notnull['avg_discount'] != 0)
airline = airline_notnull[(index1 | index2) & index3]
# print(airline.shape)

airline_selection = airline[["FFP_DATE","LOAD_TIME","LAST_TO_END","FLIGHT_COUNT","SEG_KM_SUM","avg_discount"]] 
airline_selection.head()

L= pd.to_datetime(airline_selection["LOAD_TIME"]) - pd.to_datetime(airline_selection["FFP_DATE"])
L = L.astype("str").str.split().str[0]
L = L.astype("int")/30

airline_features = pd.concat([L,airline_selection.iloc[:,2:]],axis = 1)
# print(airline_features.head())

airline_features=airline_features.rename(columns={0:'L'})
# airline_features.head()
# airline_features.describe()

airline_features_scaled= (airline_features - airline_features.mean()) / (airline_features.std())
# airline_features_scaled.head()
k = 5

kmeans_model = KMeans(n_clusters = k,random_state=123)
fit_kmeans = kmeans_model.fit(airline_features_scaled)   

r1 = pd.Series(kmeans_model.labels_).value_counts()
print(r1)
# kmeans_model.labels_[:20]
# kmeans_model.cluster_centers_ 


