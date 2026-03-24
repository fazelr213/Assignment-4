from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to range 0,1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to include channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build the neural network model
neural_network_model = Sequential()
neural_network_model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
neural_network_model.add(MaxPooling2D((2,2)))
neural_network_model.add(Flatten())
neural_network_model.add(Dense(10, activation="softmax"))

# Compile model
neural_network_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model for 15 epochs
neural_network_model.fit(X_train, y_train, epochs=15, verbose=1)

# Predict values
neural_network_model_y_pred_prob = neural_network_model.predict(X_test)
neural_network_model_y_pred = np.argmax(neural_network_model_y_pred_prob, axis=1)

# Test accuracy
print(f"Test Accuracy: {accuracy_score(y_test, neural_network_model_y_pred)}")

# CNNs are better for images because they look for patterns in nearby pixels
# Fully connected networks treat every pixel more separately and use many more parameters
# The convolution layer learns useful image features like edges, shapes, and simple clothing patterns