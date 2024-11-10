#gradient and adaboost
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gb_classifier.fit(X_train, y_train)

gb_predictions = gb_classifier.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_predictions)
print("Gradient Boosting Accuracy:", gb_accuracy)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
adaboost_classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)

gb_classifier.fit(X_pca, y)
adaboost_classifier.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
Z_gb = gb_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z_gb = Z_gb.reshape(xx.shape)
plt.contourf(xx, yy, Z_gb, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Gradient Boosting Decision Boundaries')

plt.subplot(1, 2, 2)
Z_adaboost = adaboost_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z_adaboost = Z_adaboost.reshape(xx.shape)
plt.contourf(xx, yy, Z_adaboost, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('AdaBoost Decision Boundaries')
plt.show()
