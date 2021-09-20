## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./satislar.csv') 

aylar = dataset[['Aylar']]
satislar = dataset[['Satislar']]


## Veriyi eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

## Öznitelik Ölçekleme (Sayısal verileri ölçekleme)
# from sklearn.preprocessing import StandardScaler
# standart_scaler = StandardScaler()
# X_train = standart_scaler.fit_transform(x_train)
# X_test = standart_scaler.fit_transform(x_test)
# Y_train = standart_scaler.fit_transform(y_train)
# Y_test = standart_scaler.fit_transform(y_test)


## Regresyon modeli inşaa
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()

## X_train'den Y_train'i tahmin
linear_regression.fit(x_train,y_train)

predict = linear_regression.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,linear_regression.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel('Aylar')
plt.ylabel('Satışlar')