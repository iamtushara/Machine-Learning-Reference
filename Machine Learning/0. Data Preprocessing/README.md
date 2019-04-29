# Data Pre-Processing

Before we can start building models, we need to ensure that the data passes through a certain checklist. The pre-processing phase deals with structuring the data in a way that can lead to better insights and superior model performance. This is one of the most critical and time-consuming part of any Data Science project.

## What does Data Pre-Processing entail?

* **Handling Missing Values**
Here we make strategies to handle missing values present in our dataset

* **Handling Categorical Features**
  * **Label Encoding**
  Label Encoding provides categorical features with numerical labels
  
  * **One Hot Encoding**
  In order to avoid the induced Ordinality in a Nominal data after Label Encoding, One Hot Encoding provides a set of dummy variables (equal to the number of unique values in the dependent variable) to be used in lieu of the original categorical column

* **Feature Scaling**
In order to ensure that a column does not overshadow other columns by the virtue of relatively larger numbers present within the said column compared to the other column(s)
