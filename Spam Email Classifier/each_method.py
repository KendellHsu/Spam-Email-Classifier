# each_method.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_naive_bayes(X_train_features, Y_train):
    model_NB = MultinomialNB()
    model_NB.fit(X_train_features, Y_train)
    return model_NB

def train_svc(X_train_features, Y_train):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train_features, Y_train)
    return svclassifier

def train_logistic_regression(X_train_features, Y_train):
    model_regression = LogisticRegression()
    model_regression.fit(X_train_features, Y_train)
    return model_regression

def train_random_forest(X_train_features, Y_train):
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train_features, Y_train)
    return model_rf
