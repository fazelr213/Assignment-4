from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Train the Decision Tree
breast_cancer_decision_tree = DecisionTreeClassifier(criterion="entropy")
breast_cancer_decision_tree.fit(X_train, Y_train)

# Predict values based on the training set and testing set
decision_tree_y_training_predict = breast_cancer_decision_tree.predict(X_train)
decision_tree_y_testing_predict = breast_cancer_decision_tree.predict(X_test)

# Compute accuracy of testing and training set
decision_tree_y_training_acc = accuracy_score(Y_train, decision_tree_y_training_predict)
decision_tree_y_testing_acc = accuracy_score(Y_test, decision_tree_y_testing_predict)

print(f"Training accuracy: {decision_tree_y_training_acc}")
print(f"Testing accuracy: {decision_tree_y_testing_acc}")

# Entropy measures the level of uncertainty in the data and uses that information to make decisions to separate classes more clearly
# If the training accuracy is much higher than the test accuracy, that suggests the model has overfitted the data because the model learned from the training data to well and does not generalize to unseen data well
# If the testing and training accuracy are close and high that means the model has good generalization

