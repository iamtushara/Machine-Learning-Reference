# Model Selection

This section will introduce you to concepts like Grid Search and k-Fold Cross Validation. These are techniques to analyse and improve upon the performance of the model we already built. 

**Grid Search** is used to choose amongst a list of potential values, the best combination of values to be taken as the hyper-parameter settings. This technique albeit brute, is an efficient way of finding the best set of hyper-parameters for enhancing the performance of the model we just made.

**k-Fold Cross Validation** is a better technique to evaluate the performance of a classification model than a confusion matrix. It divides the dataset into 'k' folds and makes 'k' models, each model making use of different (k-1) chunks of data for training and remaining one chunk for testing. Each model uses a different testing chunk. At the end, the mean accuracy of the k models is calculated. This is a good technique to evaluate the actual performance of a model.
