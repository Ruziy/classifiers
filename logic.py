import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('https://raw.githubusercontent.com/sdukshis/ml-intro/master/datasets/Davis.csv')

# Предобработка данных

# Разделение данных на признаки и целевую переменную
X = data.drop('sex', axis=1)
y = data['sex']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Обучение логистической модели
model = LogisticRegression()
model.fit(X_train_imputed, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_imputed)

lg_acc = accuracy_score(y_test, y_pred)
print(f"Logic accuracy: {lg_acc:.5}")

#Рисование графика 
colors = data['sex'].map({'M': 'r', 'F': 'b'})
plt.scatter(x=data['weight'], y=data['height'], c=colors)

plt.show()


