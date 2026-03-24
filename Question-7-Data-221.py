from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

# Class names for Fashion MNIST
fashion_class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Confusion matrix
CNN_confusion_mat = confusion_matrix(y_test, neural_network_model_y_pred)
print("Confusion Matrix")
print(CNN_confusion_mat)

# Find misclassified images
misclassified_iamges = np.where(neural_network_model_y_pred != y_test)[0]

# Show at least 3 misclassifed images
plt.figure(figsize=(10, 3))

for i in range(3):
    idx = misclassified_iamges[i]

    plt.subplot(1, 3, i +1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"True: {fashion_class_names[y_test[idx]]}\nPred: {fashion_class_names[neural_network_model_y_pred[idx]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# One pattern is that the model amy confuse similar clothing items, such as shirts, pullovers and coats.

# One way to improve the CNN is to add more layers for train for more epochs so the model can learn better image features