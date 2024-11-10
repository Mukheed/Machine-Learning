#Neural networks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
iris = load_iris()
X, y = iris.data, iris.target
enc = OneHotEncoder(categories='auto')
y = enc.fit_transform(y[:, np.newaxis]).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(y_test, axis=1), predicted_classes))