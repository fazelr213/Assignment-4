from sklearn.datasets import load_breast_cancer
from collections import Counter

# Load the Dataset
dataset_breast_cancer = load_breast_cancer()

# Construct feature matrix X and target Vector Y
feature_breast_cancer_X = dataset_breast_cancer.data
target_breast_cancer_Y = dataset_breast_cancer.target

# Shapes of X and Y
print(f"Shape of X: {feature_breast_cancer_X.shape}")
print(f"Shape of Y: {target_breast_cancer_Y.shape}")

# Number of samples in each class
class_counts_breast_cancer = Counter(target_breast_cancer_Y)
print(f"Class counts: {class_counts_breast_cancer}")
print(f"Target names: {dataset_breast_cancer.target_names}")

# The dataset has more benign cases than malignant cases
# This means the dataset is slightly imbalanced
# Class balance is important so it can decrease the amount of balance towards classes

