import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd

ds = pd.read_csv('Fashion-mnist_train.csv')
y = ds['label'].values
X = ds.drop('label', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# LOGISTIC========================================================================
# model = LogisticRegression(max_iter=5000)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# res = accuracy_score(y_test,y_pred)
# print(res)
# CATBOOST========================================================================
# model = CatBoostClassifier(iterations=20, eval_metric='Accuracy')
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# res = accuracy_score(y_test,y_pred)
# print(res)
# AI========================================================================
# x_train = X_train.reshape(-1, 28*28)
# x_test = X_test.reshape(-1, 28*28)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# batch_size = 128
# num_classes = 10
# epochs = 20
# model = Sequential()
# model.add(Dense(128, use_bias=False, activation='sigmoid', input_shape=(x_train.shape[1],)))
# model.add(Dense(num_classes, use_bias=False, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# print(max(history.history['val_accuracy']))
# AI_SVERT========================================================================
x_train = X_train.astype('float32').reshape(-1, 28, 28, 1) / 255
x_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255
batch_size = 128
num_classes = 10
epochs = 20
cnn_model = Sequential()
cnn_model.add(Input(shape=x_train.shape[1:], name="input"))
cnn_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
# cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
# cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes, activation='softmax', name="output"))
cnn_model.summary()
cnn_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
cnn_history = cnn_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
print(max(cnn_history.history['val_accuracy']))