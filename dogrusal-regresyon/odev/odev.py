## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

## Verilerin import edilmesi
dataset = pd.read_csv('./odev.csv') 

## veri on isleme
from sklearn import preprocessing

outlook  = dataset.iloc[:,0:1].values
windy    = dataset.iloc[:,3:4].values
temp     = dataset.iloc[:,1:2].values
humidity = dataset.iloc[:,2:3].values

play = dataset.iloc[:,-1:].values


# ENCODING
label_encoder = preprocessing.LabelEncoder()
windy[:,0] = label_encoder.fit_transform(dataset.iloc[:,3:4])
outlook[:,0] = label_encoder.fit_transform(dataset.iloc[:,0])
play[:,0] = label_encoder.fit_transform(dataset.iloc[:,-1])



# ülke verisini 1 sütundan 3 farklı sütuna ayırma
one_hot_encoder = preprocessing.OneHotEncoder()

windy = one_hot_encoder.fit_transform(windy).toarray()
play = one_hot_encoder.fit_transform(play).toarray()
outlook = one_hot_encoder.fit_transform(outlook).toarray()


## VERİLERİ PANDAS DATAFRAMELERİNE ÇEVİRME
outlook_df = pd.DataFrame(data=outlook,index=range(14),columns= ['overcast','rainy','sunny'])
windy_df = pd.DataFrame(data=windy[:,1:],index=range(14),columns= ['windy'])
temp_df = pd.DataFrame(data=temp,index=range(14),columns= ['tempeture'])
play_df = pd.DataFrame(data=play[:,1:],index=range(14),columns= ['play'])
humidity_df = pd.DataFrame(data=humidity,index=range(14),columns= ['humidity'])

final_df = pd.concat([outlook_df,temp_df,windy_df,play_df],axis=1)


# # ## Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(final_df,humidity_df,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)

y_predict = linear_regressor.predict(x_test)


# print(model.summary())

import statsmodels.api as sm

# Sabit değer ekleme
new_frame = np.append(arr=np.ones((14,1)).astype(int),values=final_df,axis=1)
new_framedf = pd.DataFrame(data=new_frame,index=range(14),columns= ['fixed_value','overcast','rainy','sunny','tempeture','windy','play'])

new_framelist = new_framedf.iloc[:,[0,1,2,3,4,5,6]].values
new_framelist = np.array(new_framelist,dtype=float)
model = sm.OLS(humidity,new_framelist).fit()



print(model.summary())




