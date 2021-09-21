## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('./maaslar.csv') 

x = dataset.iloc[:,1:2]
y = dataset.iloc[:,2:]

#linear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

x_poly = poly_reg.fit_transform(x)
regressor2 = LinearRegression()

regressor2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor2.predict(x_poly))

## tahminler

print(regressor2.predict(poly_reg.fit_transform([[11]])))
print(regressor2.predict(poly_reg.fit_transform([[6.6]])))