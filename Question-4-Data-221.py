from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# load the dataset
dataset_breast_cancer = load_breast_cancer()

# Construct feature matrix X and target Vector Y
feature_breast_cancer_X = dataset_breast_cancer.data
target_breast_cancer_Y = dataset_breast_cancer.target

# Split the data into training and testing sets using a 80/20 split with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    feature_breast_cancer_X,
    target_breast_cancer_Y,
    test_size=0.2,
    stratify=target_breast_cancer_Y,
    random_state=42
)

# Standardize the data
scaler_breast_cancer = StandardScaler()
X_train_scaled = scaler_breast_cancer.fit_transform(X_train)
X_test_scaled = scaler_breast_cancer.transform(X_test)

# Make and build the neural network model
neural_network_breast_cancer = Sequential()
neural_network_breast_cancer.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
neural_network_breast_cancer.add(Dense(1, activation="sigmoid"))

# Compile the model
neural_network_breast_cancer.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
neural_network_breast_cancer.fit(X_train_scaled, Y_train, epochs=50)

# Compute Predictions
neural_network_breast_cancer_y_train_pred_prob = neural_network_breast_cancer.predict(X_train_scaled)
neural_network_breast_cancer_y_test_pred_prob = neural_network_breast_cancer.predict(X_test_scaled)

# Convert Probability's into 1 and 0 so accuracy_score can use it
neural_network_breast_cancer_y_test_pred = (neural_network_breast_cancer_y_test_pred_prob > 0.5).astype(int).ravel()
neural_network_breast_cancer_y_train_pred = (neural_network_breast_cancer_y_train_pred_prob > 0.5).astype(int).ravel()


print(f"Training Accuracy: {accuracy_score(Y_train, neural_network_breast_cancer_y_train_pred)}")
print(f"Testing Accuracy: {accuracy_score(Y_test, neural_network_breast_cancer_y_test_pred)}")

# Scaling helps the model train better
# Sigmoid gives an output between 0 and 1
# We use 0.5 as the cutoff to choose the class
# An epoch is one full pass through the training data