#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
indices = np.random.choice(len(X), size=10, replace=False)
X_subset = X[indices]
y_subset = y[indices]
model = LogisticRegression()
model.fit(X_subset, y_subset)
plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=plt.cm.Paired)
coef = model.coef_[0]
intercept = model.intercept_
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
plt.plot([x_min, x_max], [-(coef[0] * x_min + intercept) / coef[1], -(coef[0] * x_max + intercept) / coef[1]], color='black', linestyle='--', linewidth=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()