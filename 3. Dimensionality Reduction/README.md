# Dimensionality Reduction

Dimensionality Reduction is an Unsupervised Learning technique used to reduce the number of dimensions in the dataset. Having higher number of dimensions can lead to multiple problems.
* Higher dimensions (more than 3) are impossible to visualise.
* Very computationally intensive
* Not all features are important or have actual impact on the outcome
* Curse of Dimentionalty can lead to Underflow problem while implementing Bayesian Probabilities

## Feature Selection

Feature Selection technique chooses good estimators/regressors amongst all the features present in the dataset and tries to discard the ones that do not have any significant impact on the outcome. Feature Selection can be done in one of the following ways (Backward Elimination being the most widely used):
* Backward Elimination
* Forward Selection
* Bidirectional Elimination
* Score Comparision

## Feature Extraction

Feature Extraction is used to engineer new features from the existing feature set using different statistical techniques. Some of the most common techniques to engineer new features and reduce dimentionality are:
* Principal Component Analysis
* Linear Discriminant Analysis
