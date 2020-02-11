import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# use fashion_mnist datasets for training the neural network model
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# class names for representing the labels(0-9) to string for us to understand
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# shrink the data from 0-255 to 0-1, to make it easier to compute
train_images = train_images/255.0
test_images = test_images/255.0

# show the data
# plt.imshow(train_images[7])
# plt.show()

# neural network model, input 28x28 nodes, output 10 nodes, hidden layer 128 nodes
# flatten means the array/list is flatten to be a single list, not multiple list
# dense means all nodes are connected to all nodes from the side
# relu means rectified linear unit
# softmax means all output summed is 1
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

# compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

# train data input for the model
# epochs means how many times a certain images are shown
model.fit(train_images, train_labels, epochs = 5)

# test the model with test images
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Acc: ", test_acc)

# use the model to prdecit test_images value prediction
prediction = model.predict(test_images)

# to predict 5 times
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()
