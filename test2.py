from keras.datasets import mnist
from keras import Sequential
from keras import layers
from keras.optimizers import Adam

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten data
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

print(X_train.shape, X_test.shape)

# Initialize neural network
model = Sequential(
    [
        layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile neural network
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train neural network
model.fit(X_train, y_train, epochs=3)

# Evaluate neural network
ev = model.evaluate(X_test, y_test)
print(ev)
# Predict
y_pred = model.predict(X_test)

# Save model
model.save("model.h5")
