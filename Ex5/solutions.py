import unittest
import pandas as pd
from sklearn import neighbors, svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB, MultinomialNB

class Classifier:
    def __init__(self, fName):
        self.fName = fName
        self.gaussianNB = GaussianNB()
        self.multinomialNB = MultinomialNB()
        self.knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
        self.svm = svm.SVC(kernel ='linear', C=1, gamma =1)
        self.decisionTree = tree.DecisionTreeClassifier()
        self.logisticReg = LogisticRegression()

    # Load data to variables
    def loadData(self):
        data = pd.read_csv(self.fName)
        self.Y = data['type'].as_matrix()
        self.X = data.drop('type', 1).drop('animalName', 1).applymap(lambda x: 1 if x else 0).as_matrix()

    # Split data, 1/3 to for testing, 2/3 for training
    def splitData(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = 0.333)

    # Use training data for testing
    def useFullData(self):
        self.X_train = self.X
        self.Y_train = self.Y
        self.X_test = self.X
        self.Y_test = self.Y

    # Return predicted labels for clf
    def getPrediction(self, clf):
        clf.fit(self.X_train, self.Y_train)
        return clf.fit(self.X_train, self.Y_train).predict(self.X_test)

    # Return classifier accuracy score
    def getTrainingAccuracy(self, clf):
        clf.fit(self.X_train, self.Y_train)
        return clf.score(self.X_train, self.Y_train)

    # Return cross validation score with 10-folds
    def getCrossValScore(self, clf):
        clf.fit(self.X_train, self.Y_train)
        return cross_val_score(clf, self.X_train, self.Y_train, cv=10)

    # Return cross validation test predictions
    def getCrossValPredictions(self, clf):
        return cross_val_predict(clf, self.X_train, self.Y_train, cv=10)

    # Return confusion matrix of clf
    def getConfusionMatrix(self, clf):
        self.Y_pred = clf.fit(self.X_train, self.Y_train ).predict(self.X_test)
        return confusion_matrix(self.Y_test, self.Y_pred)

    # Return Precision recall details
    def getPreciRecallDetails(self, clf):
        target_names = ['setosa', 'versicolor', 'virginica']
        self.Y_pred = clf.fit(self.X_train, self.Y_train).predict(self.X_test)
        return classification_report(self.Y_test, self.Y_pred, target_names = target_names)

###########################################################
######################### Testing #########################
###########################################################
class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.clf = Classifier('test')
        self.clf.X = pd.DataFrame({
            'a' : [1, 2, 3, 4, 5, 6],
            'b' : [1, 0, 0, 1, 0, 1]
        }).as_matrix()
        self.clf.Y = pd.DataFrame({
            't' : [1, 0, 1, 1, 1, 0]
        }).as_matrix()

    def testSplitData(self):
        self.clf.splitData()
        result = len(self.clf.X_test) == 2 and len(self.clf.X_train) == 4 and len(self.clf.Y_test) == 2 and len(self.clf.Y_train) == 4
        self.assertTrue(result)

# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)

###########################################################################################

clf = Classifier('zooData.csv')

###########################################################################################
######################  (1)
clf.loadData()

###########################################################################################
#####################   (2)

clf.useFullData() # Use full data set for testing and training
######################################
## Option 1

print 'Testing option 1 (Full training data set)....'
print 'Correct labels : '
print clf.Y_test
print ''
print ''

print 'Gaussian Naive Bayes'
print clf.getPrediction(clf.gaussianNB)
print ''

print 'Multinomial Naive Bayes'
print clf.getPrediction(clf.multinomialNB)
print ''

print 'Nearest Neighbour'
print clf.getPrediction(clf.knn)
print ''

print 'Support Vector Machine'
print clf.getPrediction(clf.svm)
print ''
print '####################################'
print ''

########################################
## Option 2

clf.splitData() # Split the data in 2/3, 1/3 ratio
print 'Testing option 2 (Split data)....'
print 'Correct labels : '
print clf.Y_test
print ''
print ''

print 'Gaussian Naive Bayes'
print clf.getPrediction(clf.gaussianNB)
print ''

print 'Multinomial Naive Bayes'
print clf.getPrediction(clf.multinomialNB)
print ''

print 'Nearest Neighbour'
print clf.getPrediction(clf.knn)
print ''

print 'Support Vector Machine'
print clf.getPrediction(clf.svm)

#############################################
## Option 3

print 'Testing option 3 (Cross validation)....'
print 'Correct labels : '
print clf.Y_test
print ''
print ''

print 'Gaussian Naive Bayes'
print clf.getCrossValPredictions(clf.gaussianNB)
print ''

print 'Multinomial Naive Bayes'
print clf.getCrossValPredictions(clf.multinomialNB)
print ''

print 'Nearest Neighbour'
print clf.getCrossValPredictions(clf.knn)
print ''

print 'Support Vector Machine'
print clf.getCrossValPredictions(clf.svm)
print ''
print '################################'

###############################################
## Decision Tree accuracy
print 'Decision Tree Training Accuracy :'
print ''

# Option 1
clf.useFullData()
print 'Decision Tree Training Accuracy : Option 1 (Full training data set) : ',clf.getTrainingAccuracy(clf.decisionTree)

# Option 2
clf.splitData()
print 'Decision Tree Training Accuracy : Option 2 (Split data) : ',clf.getTrainingAccuracy(clf.decisionTree)

# Option 3
print 'Decision Tree Training Accuracy : Option 3 (Cross Validation) : ',clf.getCrossValScore(clf.decisionTree).mean()

print ''
print '#####################################'
print ''
########################################################################################################
################################  (3)
print 'Confusion Matrices'
print ''

print 'Confusion matrix : Gaussian Naive Bayes'
print clf.getConfusionMatrix(clf.gaussianNB)
print ''

print 'Confusion matrix : Multinomial Naive Bayes'
print clf.getConfusionMatrix(clf.multinomialNB)
print ''

print 'Confusion matrix : Nearest Neighbour'
print clf.getConfusionMatrix(clf.knn)
print ''

print 'Confusion matrix : Support Vector Machine'
print clf.getConfusionMatrix(clf.svm)
print ''

print '##########################################################'
print ''
print '10-Fold Cross Validation Accuracy'
print ''

print 'Cross Validation Accuracy : Gaussian Naive Bayes'
print clf.getCrossValScore(clf.gaussianNB).mean()*100,'%'
print ''

print 'Cross Validation Accuracy : Multinomial Naive Bayes'
print clf.getCrossValScore(clf.multinomialNB).mean()*100,'%'
print ''

print 'Cross Validation Accuracy : Nearest Neighbour'
print clf.getCrossValScore(clf.knn).mean()*100,'%'
print ''

print 'Cross Validation Accuracy : Support Vector Machine'
print clf.getCrossValScore(clf.svm).mean()*100,'%'
print ''
print '#####################################################################'

######################################################################################################
#############  (5)
print 'Classification with Logistic Regression'
print ''

print 'Predicted labels with Logistic Regression'
print clf.getCrossValPredictions(clf.logisticReg)
print ''
print 'Cross Validation Accuracy : Logistic Regression '
print clf.getCrossValScore(clf.logisticReg).mean()*100,'%'


######################################################################################################
################# Answers
#
# 2) Cross Validation testing method will be more realistic, because for any classification problem,
#    100% accuracy is not practical. The options other than cross validation method gives 100% accuracy.
#    But the accuracy of cross validation method is not 100% which appears more realistics.
#
####################################################################################################
# 3)
#   There are 7 class labels.
#
# Confusion matrix : Gaussian Naive Bayes
#[[ 2  0  0  0  0  0  1]
# [ 0  6  0  0  0  0  0]
# [ 0  0  6  0  0  0  0]
# [ 0  0  0  1  2  0  0]
# [ 0  0  0  0  1  0  0]
# [ 0  0  0  0  0 14  0]
# [ 0  0  0  0  1  0  0]]
#
#   The diagonal elements represents correctly predicted values. Other non-zero elements
#   are wrong predictions. In above classifier (Gaussian Naive Bayes),
#
#   * One element of class 7, has been mis-labled as class 1.
#   * Two elements of class 5, has been mis-labled as class 4
#   * One element of class 5, has been mis-labled as class 7
#   * All other predictions are correct
#
# Confusion matrix : Multinomial Naive Bayes
#[[ 0  0  0  0  1  0  2]
# [ 0  6  0  0  0  0  0]
# [ 0  0  6  0  0  0  0]
# [ 0  3  0  0  0  0  0]
# [ 0  0  0  0  1  0  0]
# [ 0  0  0  0  0 14  0]
# [ 0  0  1  0  0  0  0]]
#
#   * One element of class 5, has been mis-labled as class 1.
#   * Two elements of class 7, has been mis-labled as class 1
#   * Three elements of class 2, has been mis-labled as class 4
#   * One element of class 3, has been mis-labled as class 7
#   * All other predictions are correct
#
#
# Confusion matrix : Nearest Neighbour
#[[ 2  0  0  0  0  0  1]
# [ 0  6  0  0  0  0  0]
# [ 0  0  6  0  0  0  0]
# [ 0  0  0  1  2  0  0]
# [ 0  0  0  0  1  0  0]
# [ 0  0  0  0  0 14  0]
# [ 0  0  1  0  0  0  0]]
#
#   * One element of class 7, has been mis-labled as class 1.
#   * Two elements of class 5, has been mis-labled as class 4
#   * One element of class 3, has been mis-labled as class 7
#   * All other predictions are correct
#
#
# Confusion matrix : Support Vector Machine
#[[ 2  0  0  0  0  0  1]
# [ 0  6  0  0  0  0  0]
# [ 0  0  6  0  0  0  0]
# [ 0  0  0  1  2  0  0]
# [ 0  0  0  0  1  0  0]
# [ 0  0  0  0  0 14  0]
# [ 0  0  1  0  0  0  0]]
#
#
#   * One element of class 7, has been mis-labled as class 1.
#   * Two elements of class 5, has been mis-labled as class 4
#   * One element of class 3, has been mis-labled as class 7
#   * All other predictions are correct
#
#############################################################################################
#  5)
#
#  The accuracy of cross Logistic Regression classifier is most of the time lower than
#  that of Gaussian Naive Bayes, Nearest Neighbour and Support Vector Machine classifiers.
#  But most of the time it is greater than the accuracy of Multinomial Naive Bayes.
#
#  Example:
#  10-Fold Cross Validation Accuracy
#
#  Cross Validation Accuracy : Gaussian Naive Bayes
#  97.1818181818 %
#
#  Cross Validation Accuracy : Multinomial Naive Bayes
#  91.6818181818 %
#
#  Cross Validation Accuracy : Nearest Neighbour
#  96.0909090909 %
#
#  Cross Validation Accuracy : Support Vector Machine
#  97.0909090909 %
#
#  Cross Validation Accuracy : Logistic Regression
#  92.9318181818 %
#

