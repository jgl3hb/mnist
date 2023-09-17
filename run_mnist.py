# Place this at the very top to only use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from PIL import Image
import numpy as np

# The code from your mnist.py file, for training the model
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

# Save the model
model.save('my_mnist_model.h5')

# Load the model back from the file
loaded_model = tf.keras.models.load_model('my_mnist_model.h5')

# Load and prepare the image
img = Image.open('image.png').convert('L')
img = img.resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array  # Invert color
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28)

# Make prediction
predictions = loaded_model.predict(img_array)
print("Predicted class:", np.argmax(predictions))
