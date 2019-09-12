#Practice of linear regression on Iris dataset

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
print("X.shape:{}".format(X.shape))
print("y.shape:{}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print("X_train.shape:{}".format(X_train.shape))
print("X_test.shape:{}".format(X_test.shape))
print("y_train.shape:{}".format(y_train.shape))
print("y_tst.shape:{}".format(y_test.shape))

regression = linear_model.LogisticRegression(solver='liblinear')
regression.fit(X_train, y_train)
print('Coefficients:%s,intercept%s'%(regression.coef_, regression.intercept_))
print("score on test:%.2f"%regression.score(X_test, y_test))
