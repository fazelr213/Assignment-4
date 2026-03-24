from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Train the Decision Tree
breast_cancer_decision_tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
breast_cancer_decision_tree.fit(X_train, Y_train)

# Predict values
decision_tree_y_predict = breast_cancer_decision_tree.predict(X_test)

# Compute accuracy
decision_tree_y_acc = accuracy_score(Y_test, decision_tree_y_predict)

# Print the confusion matrix
confusion_mat_breast_cancer_tree = confusion_matrix(Y_test,decision_tree_y_predict)
print("Decision Tree Confusion Matrix")
print(confusion_mat_breast_cancer_tree)

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
neural_network_breast_cancer_y_pred_prob = neural_network_breast_cancer.predict(X_test_scaled)

# Convert Probability's into 1 and 0 so accuracy_score can use it
neural_network_breast_cancer_y_pred = (neural_network_breast_cancer_y_pred_prob > 0.5).astype(int).ravel()

# Print the confusion matrix
confusion_mat_breast_cancer_neural = confusion_matrix(Y_test, neural_network_breast_cancer_y_pred)
print("Neural Network Confusion Matrix")
print(confusion_mat_breast_cancer_neural)

# I would prefer to use the neural network if it has higher test accuracy
# and makes fewer mistakes in the confusion matrix.

# The decision tree is easier to understand because you can follow
# the splits step by step, but it can still overfit the training data

# The neural network can learn more complex pattern which can improve performance, but it makes it harder to understand how it makes decisions
