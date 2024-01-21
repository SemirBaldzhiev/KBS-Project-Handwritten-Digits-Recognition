import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# 28x28 images of hand-written digits 0-9 (60,000 training, 10,000 testing)
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# Normalize data (pixel values from 0-255 to 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# resize data to 4D (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build model
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x_train.shape[1:])) # 64 neurons, 3x3 window, 28x28x1 input shape
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # 2x2 window

model.add(Conv2D(64, (3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # 2D to 1D data
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax')) # probability distribution

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.3)

model.save('handwritten_digits.model')

# Evaluate test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print ("Test loss: ", test_loss)
print ("Test accuracy: ", test_acc)

# Make predictions to show corectness of the model
predictions = model.predict(x_test)

print(predictions)
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()



