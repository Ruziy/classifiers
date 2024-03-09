import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns  # для красивого стиля графиков

DATASET_REDUCED_URL = 'https://gist.githubusercontent.com/sdukshis/931f187fe01468422517a09935e6623f/raw/0afe7d32f01df746804bf1f035c3bef1437c6555/kc_house_data_reduced.csv'
housesales = pd.read_csv(DATASET_REDUCED_URL)
housesales['price'] = housesales['price']/100000
housesales['sqft_living'] = housesales['sqft_living']/1000
n = len(housesales)
train_size = int(0.7 * n)
test_size = n - train_size
train = housesales.iloc[:train_size]
test = housesales.iloc[train_size:]
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(housesales)
x = train['sqft_living'].values.reshape(-1, 1)
y = train['price']
model = LinearRegression()
model.fit(x, y)

print("Коэффициенты модели:", model.coef_[0])  # Веса модели (slope)
print("Свободный член (intercept):", model.intercept_)  # Cвободный член


x_know = [[2]]
x_to_know = model.predict(x_know)
know = model.predict(x)
res = r2_score(y,know)
print(res,"R2")

# Создание графика
x = housesales['sqft_living'].values.reshape(-1, 1)
y = housesales['price']
predicted = model.predict(x)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Исходные данные')
plt.scatter(x_know, x_to_know, color='green',s=100)
plt.plot(x, predicted, color='red', linewidth=2, label='Предсказания модели')
plt.axvline(x=2, ymax=0.44, color='green', linestyle='--')
plt.axhline(y=5, xmax=0.55, color='red', linestyle='--')
plt.xlabel('Жилая площадь')
plt.ylabel('Цена')
plt.title('Предсказания модели и исходные данные')
plt.legend()
plt.show()