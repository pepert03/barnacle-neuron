import sklearn

# Neural network to predict handwritten digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
digits = load_digits()


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Initialize neural network
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))

# Train neural network
mlp.fit(X_train, y_train)

# Predict
y_pred = mlp.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))
