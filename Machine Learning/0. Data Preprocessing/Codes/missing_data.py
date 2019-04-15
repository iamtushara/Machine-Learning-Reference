# Data Preprocessing


# Importing the libraries
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')


# Splitting Independent features from Dependent ones
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# If you need to fill columns individually
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
'''Age Column'''
age = dataset.Age
age_filled = imputer.fit_transform(age.values.reshape(-1, 1))
dataset['Age_filled'] = age_filled
'''Salary Column'''
salary = dataset.Salary
salary_filled = imputer.fit_transform(salary.values.reshape(-1, 1))
dataset['Salary_filled'] = salary_filled


# Arrange all the columns
column_seq = ['Country', 'Age', 'Age_filled', 'Salary', 'Salary_filled', 'Purchased']
dataset = dataset[column_seq]


# Drop unnecessary columns
del dataset['Age'], dataset['Salary']
new_column_names = ['Country', 'Age', 'Salary', 'Purchased']
dataset.columns = new_column_names


# Save the file
dataset.to_csv('Data-Filled.csv', index=False)
