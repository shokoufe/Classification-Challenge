from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import loading_Input
from ConfusionMatrixPlot import plot_confusion_matrix
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# define baseline model
saveFile = 'trained_weights.h5'

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(300, input_dim=295, activation='relu'))
	model.add(Dense(150, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	if os.path.isfile(saveFile):
		model.load_weights(saveFile)
	return model
# loading train and test data
X = loading_Input.X_train
Y = loading_Input.y_train

X_test = loading_Input.X_test
Y_test = loading_Input.y_test

# Fit the model
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=20, verbose=1)

# using k-Fold cross validation
kfold = KFold(n_splits=20, shuffle=True, random_state=loading_Input.seed)

results = cross_val_score(estimator, X, y=Y, cv=kfold, verbose=1)

estimator.fit(X, Y)

estimator.model.save_weights(saveFile)

y_pred = estimator.predict(X_test)
Y_test_classified = list(map(lambda oneHot: np.argmax(oneHot), Y_test))
conf_matr = confusion_matrix(Y_test_classified,y_pred)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
print(classification_report(Y_test_classified, y_pred))

plot_confusion_matrix(conf_matr, ['A', 'B', 'C', 'D', 'E'],True, 'Confusion Matrix')




