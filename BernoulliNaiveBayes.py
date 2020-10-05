import pandas as pd

#importing dataset
datatest = pd.read_csv('test.csv')
datatraining = pd.read_csv('train.csv')

#Splitting the data into the tetsting dataset
Xtest = datatest.iloc[:,0:2].values
Ytest = datatest.iloc[:,2:3].values

#splitting the data into the training dataset
Xtrain = datatraining.iloc[:,0:2].values
Ytrain = datatraining.iloc[:,2:3].values

#Scaling all of the features for the training and testing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(Xtrain)
X_test = sc.transform(Xtest)

# Fitting the Bernoulli Naive Bayes Classifier
from sklearn.naive_bayes import BernoulliNB
bernoulliClassifier = BernoulliNB()
bernoulliClassifier.fit(X_train, Ytrain)

# Predicting the test set results
Y_pred = bernoulliClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(Ytest, Y_pred)

# Computing Classification Accuracy
from sklearn.metrics import accuracy_score
classificationAccuracy = accuracy_score(Ytest, Y_pred)
print()
print()

# Main Classification Metrics from the classifier
from sklearn.metrics import classification_report
report = classification_report(Ytest, Y_pred)
print(report)
