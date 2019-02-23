# Data Preprocessing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data-Filled.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Doing individualy (Optional)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
'''Country'''
country = dataset.Country
country_label_encoded = labelencoder.fit_transform(country)
dataset['Country_encoded'] = country_label_encoded

onehotencoder = OneHotEncoder(categorical_features = 'all')
dummy_var = onehotencoder.fit_transform(dataset.Country_encoded.values.reshape(-1,1)).toarray()
dataset['France'] = dummy_var[:, 0]
dataset['Spain'] = dummy_var[:, 2]
dataset['Germany'] = dummy_var[:, 1]


# Rearrange all columns
new_column_seq = ['France', 'Spain', 'Germany', 'Age', 'Salary', 'Purchased']
dataset = dataset[new_column_seq]
dataset.to_csv('Data-Complete.csv', index=False)