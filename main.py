import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Predict handwritten digit from an image.")
parser.add_argument("image_path", type=str, help="Path to the input image.")

args = parser.parse_args()

# Load model
model = load_model("handwritten_digits.model")


def predict(image_path):
    # Open the image file
    img = Image.open(image_path).convert("L")

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Convert the image to numpy array
    img = img_to_array(img)

    # Normalize the image (same as in your training)
    img = tf.keras.utils.normalize(img, axis=1)

    # plt.imshow(img)
    # plt.show()

    # Reshape the data to have 4 dimensions (batch, height, width, channels)
    img = img.reshape(-1, 28, 28, 1)

    # Predict the class using your model
    prediction = model.predict(img)

    # The predicted class
    digit = np.argmax(prediction)

    return digit


# Test the function with an image
print(predict(args.image_path))
