# Multiple Linear Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
state = dataset.State  # Taking the categorical column and label encoding it
dataset['State_Encoded'] = labelencoder.fit_transform(state.values.reshape(-1, 1))
onehotencoder = OneHotEncoder(categorical_features = 'all')
ohe = onehotencoder.fit_transform(dataset['State_Encoded'].values.reshape(-1, 1)).toarray()
new_columns = list(state.sort_values().unique())
for index, column in enumerate(new_columns):
	dataset[column] = ohe[:,index]
dataset = dataset.iloc[:, [0,1,2,6,7,8,4]]  # Re-arranging the required columns
del ohe, state, column, index, new_columns

# Splitting the Independent and Dependent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Avoiding the Dummy Variable Trap (Optional, the library already does this)
"""X = X[:, :-1]"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
# The library is same as it was for Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)