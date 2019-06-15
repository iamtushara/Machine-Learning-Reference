# Xtreme Gradient Boosting Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rfc_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc_classifier.fit(X_train, y_train)
rfc_y_pred = rfc_classifier.predict(X_test)

# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)

# Fitting XGBoost Classifier to the Training Set
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
xgb_y_pred = xgb_classifier.predict(X_test)

# Making the Confusion Matrices
from sklearn.metrics import confusion_matrix
rfc_cm = confusion_matrix(y_test, rfc_y_pred)
nb_cm = confusion_matrix(y_test, nb_y_pred)
xgb_cm = confusion_matrix(y_test, xgb_y_pred)

# Performing K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
nb_cv = cross_val_score(estimator = nb_classifier, X=X_train, y=y_train, cv=10)
rfc_cv = cross_val_score(estimator = rfc_classifier, X=X_train, y=y_train, cv=10)
xgb_cv = cross_val_score(estimator = xgb_classifier, X=X_train, y=y_train, cv=10)
cvs = [nb_cv, rfc_cv, xgb_cv]


# Classification Modelling Analysis
accuracies = pd.DataFrame(columns = ['Naive Bayes', 'Random Forest', 'XGBoost'],
			  index = ['Accuracy', 'CV Mean', 'CV Std'])
confusion_matrices = [nb_cm, rfc_cm, xgb_cm]
for index, cm in enumerate(confusion_matrices):
	accuracies.iloc[0, index] = cm.diagonal().sum() * 100 / cm.sum()
	accuracies.iloc[1, index] = cvs[index].mean()
	accuracies.iloc[2, index] = cvs[index].std()
