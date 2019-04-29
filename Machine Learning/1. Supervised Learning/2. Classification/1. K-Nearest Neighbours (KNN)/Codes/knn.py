# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Visualising the dataset
plt.figure(figsize=(18,12))
sns.set_context('poster', font_scale=0.5)
sns.scatterplot(x='Age', y='EstimatedSalary', style='Gender', hue='Purchased', data=dataset, legend='full')
plt.show()


# Splitting the Independent and Dependent features
X = dataset.iloc[:, [2, 3]].values  # Using only Age and Gender columns
y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, linewidths=0.2, cmap='Set1')

# Printing the accuracies
print("Total rows = {}".format(cm.sum()))
print("Correct Answers = {}".format(cm.diagonal().sum()))
print("Accuracy = {}%".format((cm.diagonal().sum()/cm.sum()) * 100))

# Asking questions from the model
questions = np.array([[40, 50000], [35, 100000]])
questions_transformed = sc.transform(questions)
answers = classifier.predict(questions_transformed)

# Finding the optimal number of neighbours
accuracies = []
for n in range(1,16,2):
	classifier = KNeighborsClassifier(n_neighbors=n)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	accuracies.append(cm.diagonal().sum() * 100 / cm.sum())

plt.plot(range(1,16,2), accuracies)
plt.show()
