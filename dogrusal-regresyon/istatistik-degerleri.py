## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

## Verilerin import edilmesi
veriler = pd.read_csv('./veriler.csv') 


#yaş ile alakalı olan sütunu seçiyoruz
age = veriler.iloc[:,3:4].values 

boy_kilo = veriler.iloc[:,1:3].values
gender = veriler.iloc[:,-1:].values


## ÜLKE VERİSİNİ İŞLEME

country = veriler.iloc[:,0:1].values

## veri on isleme
from sklearn import preprocessing

# ENCODING
label_encoder = preprocessing.LabelEncoder()
gender[:,-1] = label_encoder.fit_transform(veriler.iloc[:,-1])
country[:,0] = label_encoder.fit_transform(veriler.iloc[:,0])

# ülke verisini 1 sütundan 3 farklı sütuna ayırma
one_hot_encoder = preprocessing.OneHotEncoder()
gender = one_hot_encoder.fit_transform(gender).toarray()
country = one_hot_encoder.fit_transform(country).toarray()


# ## VERİLERİ PANDAS DATAFRAMELERİNE ÇEVİRME
country_df = pd.DataFrame(data=country,index=range(22),columns= ['fr','tr','us'])
age_df = pd.DataFrame(data=age,index=range(22),columns=['yas'])
b_k_df = pd.DataFrame(data=boy_kilo,index=range(22),columns=['boy','kilo'])
gender_df = pd.DataFrame(data=gender[:,:1],index=range(22),columns=['cinsiyet'])

# ## VERİLERİ BİRLEŞTİRME
final_df = pd.concat([country_df,b_k_df,age_df],axis=1)

final_df2 = pd.concat([final_df,gender_df],axis=1)


# ## Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(final_df,gender_df,test_size=0.33,random_state=0)

# """
# x: bağımsız değişken
# y: bağımlı değişken
# """

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)

y_predict = linear_regressor.predict(x_test)


height = final_df2.iloc[:,3:4].values

left = final_df2.iloc[:,:3]
right = final_df2.iloc[:,4:]

dataset = pd.concat([left,right],axis=1)

x_train,x_test,y_train,y_test = train_test_split(dataset,height,test_size=0.33,random_state=0)

linear_regressor2 = LinearRegression()
linear_regressor2.fit(x_train,y_train)

y_predict2 = linear_regressor2.predict(x_test)


### İSTATİSTİK DEĞERLERİ - BACKWARD ELIMINATION


import statsmodels.api as sm

# Sabit değer ekleme
new_frame = np.append(arr=np.ones((22,1)).astype(int),values=dataset,axis=1)

new_framelist = dataset.iloc[:,[0,1,2,3,4,5]].values
new_framelist = np.array(new_framelist,dtype=float)
model = sm.OLS(height,new_framelist).fit()

print(model.summary())


