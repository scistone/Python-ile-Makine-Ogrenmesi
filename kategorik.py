## Kütüphaneler
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

## Verilerin import edilmesi
veriler = pd.read_csv('./veriler.csv') 


# Ülke verisini çekmek
country = veriler.iloc[:,0:1].values

"""
iloc iki adet parametre almaktadır. 
birincisi: hangi satırdan hangi satıra kadar veriyi okuyacağı
ikincisi:  hangi sütundan hangi sütuna kadar veriyi okuyacağı
"""


## veri on isleme
from sklearn import preprocessing

# ENCODING
label_encoder = preprocessing.LabelEncoder()
country[:,0] = label_encoder.fit_transform(veriler.iloc[:,0])

# ülke verisini 1 sütundan 3 farklı sütuna ayırma
one_hot_encoder = preprocessing.OneHotEncoder()
country = one_hot_encoder.fit_transform(country).toarray()