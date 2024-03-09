import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
# Загрузка данных
data = pd.read_csv('https://raw.githubusercontent.com/sdukshis/ml-intro/master/datasets/Davis.csv')

# Предобработка данных

# Разделение данных на признаки и целевую переменную
X = data.drop('sex', axis=1)
y = data['sex']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

catbst = CatBoostClassifier(iterations=20, eval_metric='Accuracy')
catbst.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False, plot=False)

y_pred_cb = catbst.predict(X_test)
cb_acc = accuracy_score(y_test, y_pred_cb)
print(f"CatBoost accuracy: {cb_acc:.5}")
plt.show()