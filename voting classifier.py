#voting classifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data[:, :4]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
estimators = []
estimators.append(('LR', LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)))
estimators.append(('SVC', SVC(gamma='auto', probability=True)))
estimators.append(('DTC', DecisionTreeClassifier()))
vot_hard = VotingClassifier(estimators=estimators, voting='hard')
vot_hard.fit(X_train, y_train)
y_pred_hard = vot_hard.predict(X_test)
hard_voting_score = accuracy_score(y_test, y_pred_hard)
print("Hard Voting Score: %.3f" % hard_voting_score)
vot_soft = VotingClassifier(estimators=estimators, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred_soft = vot_soft.predict(X_test)
soft_voting_score = accuracy_score(y_test, y_pred_soft)
print("Soft Voting Score: %.3f" % soft_voting_score)