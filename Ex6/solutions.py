import unittest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score

class FeatureSelection:
    def __init__(self, fName):
        self.fName = fName
        self.knn = neighbors.KNeighborsClassifier(n_neighbors = 1)

    # Load data to variables
    def loadData(self):
        data = pd.read_csv(self.fName)
        self.X_train = data.drop('Class', 1)
        self.Y_train = data['Class']

    # Return cross validation score with 10-folds
    def getCrossValScore(self, clf, X, Y):
        clf.fit(X, Y)
        return cross_val_score(clf, X, Y, cv=10)

    # Return first n principal components
    def getFirstPCs(self, n):
        pca = PCA()
        pca.fit(self.X_train)
        X = pca.transform(self.X_train)
        return pd.DataFrame(X[:,:n])

    # Return data reduced to n PCs
    def getPCs(self, n):
        pca = PCA(n_components=n)
        pca.fit(self.X_train)
        return pca.transform(self.X_train)

    # Plot first two PCs
    def plotPCs(self):
        pcs = self.getFirstPCs(2)
        plt.title('PC1 vs PC2')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.scatter(pcs[0], pcs[1])
        plt.show()

    # Get n best features using chi-square method
    def getBestFeatures(self, n):
        return SelectKBest(chi2, k=n).fit_transform(self.X_train, self.Y_train)

###########################################################
######################### Testing #########################
###########################################################
class FeatureSelectionTest(unittest.TestCase):
    def setUp(self):
        self.clf = FeatureSelection('test')

# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)

###########################################################################################

obj = FeatureSelection('colonTumor.csv')

###########################################################################################
######################  (1)
obj.loadData()
print 'Nearest Neighbour Cross Validation Accuracy with training data'
print obj.getCrossValScore(obj.knn, obj.X_train, obj.Y_train).mean()

first10pcs =  obj.getFirstPCs(10) # Get first 10 PCs
print 'Nearest Neighbour Cross Validation Accuracy with first 10 PCs'
print obj.getCrossValScore(obj.knn, first10pcs, obj.Y_train).mean()
print ''
print '#########################################################################'

print 'Cross validation accuracy for KNN with 1 PC'
print obj.getCrossValScore(obj.knn, obj.getPCs(1), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 2 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(2), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 3 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(3), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 4 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(4), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 5 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(5), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 6 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(6), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 7 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(7), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 8 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(8), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 9 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(9), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 10 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(10), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 50 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(50), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with 100 PCs'
print obj.getCrossValScore(obj.knn, obj.getPCs(100), obj.Y_train).mean()
print ''
print '#########################################################################'

print 'Cross validation accuracy for KNN with the best feature'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(1), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 2 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(2), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 3 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(3), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 4 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(4), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 5 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(5), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 6 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(6), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 7 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(7), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 8 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(8), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 9 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(9), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 10 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(10), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 50 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(50), obj.Y_train).mean()
print 'Cross validation accuracy for KNN with the 100 best features'
print obj.getCrossValScore(obj.knn, obj.getBestFeatures(100), obj.Y_train).mean()
print '#########################################################################'

obj.plotPCs() # (4) Visualize first 2 PCs

############################################################################
#
# (3)
#    The accuracies increase from 1-component limit to 6-components. Then
#    the accuracies decrease upto 10-component limit.
#    For 50-component and 100-component limit, the accuracies are high and are
#    equal.
#    The accuracy decrease from 1 to 10 could be due to overfitting caused by some
#    components as the attributes weren't normalized.
#    50 and 100 component accuracies are equal to the accuracy given by considering
#    all attributes. This means that, with only 50 components, a good accuracy
#    can be obtained. Therefore no need to consider all attributes or 100 PCs.
#    With only 50 PCs, a good essence of the data can be obtained.
#
# (4)
#    The values of the plot are scattered everywhere. Therefore, the first two PCs
#    doesn't have much of a correlation. They appear to be intependent of each other.
#
# (5)
#    The accuracy values fluctuates from 0.7 to 0.88 when number of features is increased.
#    But, the accuracy gets to a peak value for number of features 9 and 10.
#    Accuracy is lower for 100 best features than for 9 best features.
#    Therefore, after some point, adding more features decreases the accuracy.
#
# (6)
#    The maximum accuracy recorded with principal components is 0.785. The accuracies for
#    different number of principal components ranged from 0.567 to 0.785
#    But with chi square feature selection method, a maximum accuracy of 0.883 was recorded.
#    The accuracies for different number of features ranged from 0.710 to 0.883.
#    The accuracy range of chi square method seems to be better.
#    Therefore, for the given data set, chi square method is more suitable.
#
#
