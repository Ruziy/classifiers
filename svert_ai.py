import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Input,Dense
import matplotlib.pyplot as plt

#Загрузка данных
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

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

plt.figure(figsize=(10, 6))
plt.plot(cnn_history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.ylim(0.91, 1.01)
plt.show()