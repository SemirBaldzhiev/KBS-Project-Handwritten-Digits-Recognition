import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.datasets import mnist


(_, _), (x_test, y_test) = mnist.load_data() 

x_test = tf.keras.utils.normalize(x_test, axis=1)

x_test = x_test.reshape(-1, 28, 28, 1)

loaded_model = load_model('handwritten_digits.model')

y_pred = loaded_model.predict(x_test)

y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f'\n\nAccuracy: {accuracy * 100:.2f}%')

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_labels)
print('\nConfusion Matrix:')
print(conf_matrix)