""" K-Nearest Neighbor Classifier"""
# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
data = pd.read_csv('CSC340GradeData.csv')

X = data.iloc[:,0:4].values
Y = data.iloc[:,4:6]

#creating dummy variable for the Binary classifier
ForP = pd.get_dummies(Y['ForP'],drop_first=True)
Y=Y.drop('ForP',axis=1)
Y=pd.concat([Y,ForP],axis=1)
Y=Y.drop('Grade', axis=1)


# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# Fitting the KNN Classifier
from sklearn.neighbors import KNeighborsClassifier 
knnClassifier = KNeighborsClassifier(n_neighbors = 5)
knnClassifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = knnClassifier.predict(X_test)


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

