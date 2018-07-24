
import Preprocessing 
from Preprocessing import *

#LOGISTIC REGRESSION


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



#Calculating the accuracy
from sklearn import metrics
acs = metrics.accuracy_score(y_test, y_pred)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

