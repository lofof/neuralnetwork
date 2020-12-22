# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio

MNIST = tf.keras.datasets.mnist
[x_train, y_train],[x_test, y_test] = MNIST.load_data()


print(x_test.shape, y_test.shape, sep = '\n')

x_train =  x_train/255
x_test = x_test/255

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28,28,1)),
      tf.keras.layers.Dense(128, activation = tf.nn.relu),
      tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

print(model.summary())

model.compile(
    
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

Model = model.fit(x_train,y_train,batch_size=32, epochs=10)

print(model.evaluate(x_test, y_test))

plt.figure(figsize=(12,12))
for i in range(36):
    plt.subplot(6,6, i+1)
    plt.xticks([])
    plt.yticks([])
    x = np.expand_dims(x_test[i], axis = 0)
    res = model.predict(x)

    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Number: {np.argmax(res)}")
plt.show()


accu_values = Model.history['accuracy']
epochs = range(1,len(accu_values) + 1)
plt.plot(epochs,accu_values,label = 'Метрика качества')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


accu_values = Model.history['loss']
epochs = range(1,len(accu_values) + 1)
plt.plot(epochs,accu_values,label = 'Функция потерь')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()



def ans(model, filename, display = True):
    img = imageio.imread(filename)
    img = np.mean(img, 2, dtype = float)
    img = img/255
    if (display):
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap = plt.cm.binary)
        plt.xlabel(filename)
        img = np.expand_dims(img, 0)
        plt.title(f"Number: {np.argmax(model.predict(img))}")
        plt.show()    
    return 

for i in range(4):
    filename = f'{i}.png'
    ans(model, filename)