
import Preprocessing 
from Preprocessing import *
 
#NAIVE BAYES ALGORITHM

# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


#Calculating the accuracy
from sklearn import metrics
acs = metrics.accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


