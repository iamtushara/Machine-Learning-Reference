# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_scaled, y)

choice = 'y'
while choice == 'y':
	age = float(input("Enter age: "))
	salary = float(input("Enter salary: "))
	questions = np.array([[age, salary]])
	questions_scaled = sc.transform(questions)
	answer = 'LIKELY' if classifier.predict(questions_scaled) else "NOT LIKELY"
	print("The customer with age {} and Salary {} is {} to buy our product".format(age, salary, answer))
	choice = input("More questions? (y/n): ")

print("End of Program")
input()

	