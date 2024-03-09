import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt

#Загрузка данных
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Нормализация по пиксилям
x_train = X_train.reshape(-1, 28*28)
x_test = X_test.reshape(-1, 28*28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Параметры модели
batch_size = 128
num_classes = 10
epochs = 20

model = Sequential()
model.add(Dense(128, use_bias=False, activation='sigmoid', input_shape=(x_train.shape[1],)))
model.add(Dense(num_classes, use_bias=False, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.ylim(0.91, 1.01)
plt.show()
