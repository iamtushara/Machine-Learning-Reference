# Importing the libraries
import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('Data.csv')


# Splitting Independent features from Dependent ones
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]


# Converting the pandas objects to numpy arrays
X_array = X.values
y_array = y.values


# Printing the count of missing values
pd.isnull(X['Country']).sum()
pd.isnull(X['Age']).sum()
pd.isnull(X['Salary']).sum()


# Importing Simple Imputer from sklearn library
from sklearn.impute import SimpleImputer


# Creating imputer objects
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')


# Filling the missing values using the imputer objects
dataset['Country_Filled'] = mode_imputer.fit_transform(dataset.Country.values.reshape(-1,1))
dataset['Age_Filled'] = mean_imputer.fit_transform(dataset.Age.values.reshape(10,1))
dataset['Salary_Filled'] = mean_imputer.fit_transform(dataset.Salary.values.reshape(-1,1))


# Keeping only the required columns in sequence
dataset = dataset[['Country_Filled', 'Age_Filled', 'Salary_Filled', 'Purchased']]
dataset.columns = ['Country', 'Age', 'Salary', 'Purchased']


# Encoding categorical data

# Importing the library to Label Encode and One-Hot Encode our categorical column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()


'''Country Column is the only categorical column in the dataset'''

# Label Encoding the Country column
dataset['Country_encoded'] = labelencoder.fit_transform(dataset.Country)


# One-Hot encoding the Country column
onehotencoder = OneHotEncoder(categorical_features = 'all')
dummy_var = onehotencoder.fit_transform(dataset.Country_encoded.values.reshape(-1,1)).toarray()


# the variable 'dummy_var' now contains an array with 3 columns 
# because there are 3 unique countries

# Lets create new columns for each of the countries
dataset['France'] = dummy_var[:, 0]
dataset['Germany'] = dummy_var[:, 1]
dataset['Spain'] = dummy_var[:, 2]

"""Use the below commented code to do the above column task automatically"""
#unique_countries = sorted(list(dataset.Country.unique()))
#for index, country in enumerate(unique_countries):
#	dataset[country] = dummy_var[:, index]


# Rearrange all columns
new_column_seq = ['France', 'Germany', 'Spain', 'Age', 'Salary', 'Purchased']
dataset = dataset[new_column_seq]


# Reassigning X and y to correspond to the new Independent & Dependent Variables
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


# Scaling back the data to the original scale using inverse_transform

scaler.mean_   # This will show the mean of all the columns
scaler.scale_  # This will show the standard-deviation of all the columns

X_train_original = scaler.inverse_transform(X_train_transformed)
X_test_original = scaler.inverse_transform(X_test_transformed)
