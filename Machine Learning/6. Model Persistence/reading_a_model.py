from joblib import load
new_classifier = load('rfc_model.joblib')

import numpy as np
questions = np.array([
		[34, 56000],
		[67, 89000]])

answers = new_classifier.predict(questions)
print(answers)


