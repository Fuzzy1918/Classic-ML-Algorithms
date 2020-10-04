# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
data = pd.read_csv('CSC340GradeData.csv')

X = data.iloc[:,0:4].values
Y = data.iloc[:,4:6]

#creating dummy variable for the Binary classifier
ForP = pd.get_dummies(Y['ForP'],drop_first=True)
Y=Y.drop('ForP',axis=1)
Y=pd.concat([Y,ForP],axis=1)
Y=Y.drop('Letter Grade', axis=1)


# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardScaler_X = StandardScaler()
X_train = standardScaler_X.fit_transform(X_train)
X_test = standardScaler_X.transform(X_test)


# Fitting Linear SVM to the training set
from sklearn.svm import SVC

# No Kernel Trick
#classifier = SVC(kernel = 'linear')



# Using RBF Kernel
classifier = SVC(kernel = 'rbf' )


classifier.fit(X_train, Y_train)




# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(Y_test, Y_pred)


# Computing Classification Accuracy
from sklearn.metrics import accuracy_score
classificationAccuracy = accuracy_score(Y_test, Y_pred)
print()
print()

# Main Classification Metrics from the classifier
from sklearn.metrics import classification_report
report = classification_report(Y_test, Y_pred)
print(report)


