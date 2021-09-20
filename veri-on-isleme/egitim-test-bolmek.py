## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

## Verilerin import edilmesi
veriler = pd.read_csv('./eksik-veriler.csv') 

# imputer ayarları burada yapılıyor
# missing values olarak nan olan değerleri gösteriyoruz
# strategy olarak da mean yani ortalama olduğunu söylüyoruz
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

#yaş ile alakalı olan sütunu seçiyoruz
age = veriler.iloc[:,3:4].values 

boy_kilo = veriler.iloc[:,1:3].values
gender = veriler.iloc[:,-1].values


# age olarak aldığımız yaş sütununu imputer'a gönderip ilk belirlediğimiz stratejiye göre öğrenme işlemini gerçekleştiriyoruz.
imputer = imputer.fit(age)

# daha sonrasında yaş sütunu için imputer.transform() metodunu çağırıp değerlerini değiştiriyoruz..
age =  imputer.transform(age)


# daha sonrasında yaş sütunu için imputer.transform() metodunu çağırıp değerlerini değiştiriyoruz..
age =  imputer.transform(age)

## ÜLKE VERİSİNİ İŞLEME

country = veriler.iloc[:,0:1].values

## veri on isleme
from sklearn import preprocessing

# ENCODING
label_encoder = preprocessing.LabelEncoder()
country[:,0] = label_encoder.fit_transform(veriler.iloc[:,0])

# ülke verisini 1 sütundan 3 farklı sütuna ayırma
one_hot_encoder = preprocessing.OneHotEncoder()
country = one_hot_encoder.fit_transform(country).toarray()


## VERİLERİ PANDAS DATAFRAMELERİNE ÇEVİRME
country_df = pd.DataFrame(data=country,index=range(22),columns= ['fr','tr','us'])
age_df = pd.DataFrame(data=age,index=range(22),columns=['yas'])
b_k_df = pd.DataFrame(data=boy_kilo,index=range(22),columns=['boy','kilo'])
gender_df = pd.DataFrame(data=gender,index=range(22),columns=['gender'])

## VERİLERİ BİRLEŞTİRME
final_df = pd.concat([country_df,b_k_df,age_df],axis=1)

final_df2 = pd.concat([final_df,gender_df],axis=1)


## Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(final_df,gender_df,test_size=0.33,random_state=0)

"""
x: bağımsız değişken
y: bağımlı değişken
"""

## Öznitelik Ölçekleme (Sayısal verileri ölçekleme)
from sklearn.preprocessing import StandardScaler
standart_scaler = StandardScaler()
X_train = standart_scaler.fit_transform(x_train)
X_test = standart_scaler.fit_transform(x_test)

