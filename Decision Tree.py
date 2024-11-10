#Decision Tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/User_Data.csv')

X = df.iloc[:, [2, 3]].values
Y = df.iloc[:, 4].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_Train, Y_Train)

plt.figure(figsize=(10, 8))
plot_tree(classifier, feature_names=['Feature 1', 'Feature 2'], class_names=['0', '1'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()

Y_Pred = classifier.predict(X_Test)

accuracy = accuracy_score(Y_Test, Y_Pred)
print(f"Accuracy: {accuracy:.2f}")
