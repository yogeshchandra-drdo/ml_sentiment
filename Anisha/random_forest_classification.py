
import Preprocessing 
from Preprocessing import *

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#RANDOM FOREST CLASSIFICATION ALGORITHM

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)



#Calculating the accuracy
from sklearn import metrics
acs = metrics.accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

