import os
import pickle

from tensorflow import keras  # noqa
import numpy as np
from matplotlib import pyplot as plt


data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# since labels are integers in the dataset, here's their text representation (by their index)
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# images are 2D arrays: 28 rows of 28 pixels, where each pixel is represented by <0 <= int <= 255>
print("Data BEFORE shrinking")
print(train_images[0][3:6])  # first image, rows 3 to 5
plt.imshow(train_images[0])
plt.title("Image BEFORE shrinking the data")
plt.show()

# it's hard to work with numbers in such a huge range, so we need to "shrink" it
train_images = train_images / 255.0  # this divides each value in all sub-arrays
test_images = test_images / 255.0  # this divides each value in all sub-arrays
# now the data range is much smaller, but the image still looks the same
print()
print("Data AFTER shrinking")
print(train_images[0][3:6])  # first image, rows 3 to 5
plt.imshow(train_images[0])
plt.title("Image AFTER shrinking the data")
plt.show()

# here's the neuron network architecture we're going to use:
# flatten 28 sub-arrays and get 784 pixels in a single array
# create the input neuron layer with 784 neurons (1 per pixel)
# create the output neuron layer with 10 neurons (1 per label)
# create 1 hidden layer between the above 2, with 128 (this number is arbitrary) neurons

model_path = os.path.join(os.path.dirname(__file__), "image_classification_model.pkl")
if os.path.isfile(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = keras.Sequential(
        layers=[
            keras.layers.Flatten(input_shape=(28, 28)),
            # Dense: each neuron or node in the layer is connected to every neuron in the previous layer
            # activation Rectified Linear Unit: ReLU(x) = max(0, x), Output Range: [0, +∞]
            keras.layers.Dense(units=128, activation="relu"),
            # activation Softmax: Output Range: [0, 1] (normalized probability distribution over multiple classes)
            keras.layers.Dense(
                units=len(class_names), activation="softmax"
            ),  # used for multi-class classification
        ]
    )

    model.compile(
        optimizer="adam",  # Adaptive Moment Estimation
        loss="sparse_categorical_crossentropy",  # used for multi-class classification where the target values are ints
        metrics=["accuracy"],  # measures the percentage of correctly classified samples
    )

    model.fit(
        x=train_images, y=train_labels, epochs=5
    )  # epochs - how many times the model is going to analyze each image

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

img_count_to_predict = 20
fig, axes = plt.subplots(nrows=1, ncols=img_count_to_predict)
for image, ax in zip(test_images[:img_count_to_predict], axes):
    image_arr = np.array([image])
    prediction = model.predict(image_arr)
    index_of_the_predicted_class = np.argmax(prediction[0])
    predicted_class = class_names[index_of_the_predicted_class]

    ax.imshow(image)
    ax.set_title(predicted_class)
plt.tight_layout()
plt.show()
