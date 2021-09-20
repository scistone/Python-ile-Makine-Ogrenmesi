#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:35:31 2020

@author: deger
"""

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

"""
iloc iki adet parametre almaktadır. 
birincisi: hangi satırdan hangi satıra kadar veriyi okuyacağı
ikincisi:  hangi sütundan hangi sütuna kadar veriyi okuyacağı
"""

# age olarak aldığımız yaş sütununu imputer'a gönderip ilk belirlediğimiz stratejiye göre öğrenme işlemini gerçekleştiriyoruz.
imputer = imputer.fit(age)

# daha sonrasında yaş sütunu için imputer.transform() metodunu çağırıp değerlerini değiştiriyoruz..
age =  imputer.transform(age)
