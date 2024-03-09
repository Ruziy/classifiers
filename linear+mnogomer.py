import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KC_HOUSE_DATA_URL = 'https://gist.github.com/sdukshis/b98e467996e6019f7858b36c2d4643bf/raw/cf999800519c8121e6ac43d7159d0a5026ab6fbf/kc_house_data.csv'
housesales = pd.read_csv(KC_HOUSE_DATA_URL)
housesales['SalePrice'] = housesales['SalePrice']/100000
housesales.head()
n = len(housesales)
m = len(housesales.drop('SalePrice', axis=1).columns)
y = housesales['SalePrice'].values.reshape((n, 1))
X = housesales.drop('SalePrice', axis=1).values.reshape((n, m))
X = np.hstack((np.ones((n, 1)), X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train,  y_train)

y_pred = model.predict(X_test)
res =r2_score(y_test,y_pred)

print (res)