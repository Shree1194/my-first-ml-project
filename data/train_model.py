# Simple ML model to predict iris flowers
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")