import pandas as pd
import math
import unittest

class Statistics() :
    def __init__(self, fileName):
        self.fileName = fileName

    #read data to Statistics object
    def readData(self):
        self.data = pd.read_csv(self.fileName, index_col=False, header=0)
        self.number_of_vectors = self.data.shape[1]
        self.elements_per_vector = self.data.shape[0]

    #fill NaN values with mean
    def preProcess(self):
        for i in range(0, self.number_of_vectors):
            self.data.iloc[:, i].fillna(self.data.iloc[:, i].mean(), inplace=True)

    #find standard deviation of a vector
    def findVariance(self, vector):
        mean = vector.mean()
        return vector.apply(lambda x: (x-mean)**2).sum()/(self.elements_per_vector-1)

    #find covariance between two vectors
    def findCovariance(self, vector1, vector2):
        v1_mean = vector1.mean()
        v2_mean = vector2.mean()
        vector1 = vector1.apply(lambda x: x-v1_mean)
        vector2 = vector2.apply(lambda x: x-v2_mean)
        return (vector1 * vector2.values).sum()/(self.elements_per_vector-1)

    #find standard deviation of a vector
    def findCorrelation(self, vector1, vector2):
        return self.findCovariance(vector1, vector2)/(math.sqrt(self.findVariance(vector1)) * math.sqrt(self.findVariance(vector2)))

    #find covariance matrix
    def getCovarianceMatrix(self):
        #Initialize matrix
        matrix = [[0 for i in xrange(self.number_of_vectors)] for i in xrange(self.number_of_vectors)]

        #Fill in variance
        for i in range(0, self.number_of_vectors):
            matrix[i][i] = self.findVariance(self.data.iloc[:,i])

        #Fill in covariance values
        for i in range(0, self.number_of_vectors):
            for j in range(i+1, self.number_of_vectors):
                element = self.findCovariance(self.data.iloc[:,i], self.data.iloc[:,j])
                matrix[i][j] = element
                matrix[j][i] = element
        return matrix

    # find correlation matrix
    def getCorrelationMatrix(self):
        # Initialize matrix
        matrix = [[0 for i in xrange(self.number_of_vectors)] for i in xrange(self.number_of_vectors)]

        # Fill in correlation values
        for i in range(0, self.number_of_vectors):
            for j in range(0, self.number_of_vectors):
                element = self.findCorrelation(self.data.iloc[:, i], self.data.iloc[:, j])
                matrix[i][j] = element
                matrix[j][i] = element
        return matrix

#######################################
###############Testing#################
#######################################
class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.statObj = Statistics("test")
        self.statObj.data = pd.DataFrame([[0.5, 0.6],
                                          [0.1, 0.9],
                                          [0.7, 0.4],
                                          [0.8, 0.3],
                                          [0.1, 0.9]])
        self.statObj.number_of_vectors = self.statObj.data.shape[1]
        self.statObj.elements_per_vector = self.statObj.data.shape[0]

    def testFindVariance(self):
        result = self.statObj.findVariance(self.statObj.data.iloc[:, 0])
        expected = self.statObj.data.iloc[:, 0].var() # Using pandas internal function
        self.assertAlmostEqual(result, expected, places=3, msg=None, delta=None)

    def testFindCovariance(self):
        result = self.statObj.findCovariance(self.statObj.data.iloc[:, 0], self.statObj.data.iloc[:, 1])
        expected = self.statObj.data.iloc[:, 0].cov(self.statObj.data.iloc[:, 1]) # Using pandas internal function
        self.assertAlmostEqual(result, expected, places=3, msg=None, delta=None)

    def testFindCorrelation(self):
        result = self.statObj.findCorrelation(self.statObj.data.iloc[:, 0], self.statObj.data.iloc[:, 1])
        expected = self.statObj.data.iloc[:, 0].corr(self.statObj.data.iloc[:, 1]) # Using pandas internal function
        self.assertAlmostEqual(result, expected, places=3, msg=None, delta=None)


# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)


obj = Statistics('data.csv')
obj.readData()
obj.preProcess()

covarianceMatrix = obj.getCovarianceMatrix()
correlationMatrix = obj.getCorrelationMatrix()

print 'Covariance Matrix\n'
for i in range(0, obj.number_of_vectors):
    print covarianceMatrix[i]

print '\nCorrelation Matrix\n'
for i in range(0, obj.number_of_vectors):
    print correlationMatrix[i]

########################################################
#(2) Below is the correlation matrix
#
# [1.0, 0.8297978860185552, 0.79625518304020937, 0.79328108801390362, 0.41765589960540889]
# [0.8297978860185552, 1.0, 0.85784223599507503, 0.84039895216522742, 0.56335341628494839]
# [0.79625518304020937, 0.85784223599507503, 1.0000000000000002, 0.90305086143141644, 0.60851499129083564]
# [0.79328108801390362, 0.84039895216522742, 0.90305086143141644, 1.0000000000000002, 0.72772284237180351]
# [0.41765589960540889, 0.56335341628494839, 0.60851499129083564, 0.72772284237180351, 1.0]
#
# All the values are non-zero. Therefore there are no uncorrelated vectors in the given data set.
# The diagonal elements represents the correlation of vectors between themselves. The diagonal elements
# take the value 1 which means that they are fully correlated.
# The correlation value ranges from 0 to 1. The columns with correlation value close to 1 are highly
# correlated while the ones with values close to 0 are less correlated.

#(3) Below is the condition used
#
#   (mean(column1) + mean(column5))/2 > (mean(column1) + mean(column2) + mean(column3))/3
#
#   Although mean values can give some idea about the nature of the vectors,
#   correlation or covariance values gives much better idea when multiple vectors
#   are involved. Correlation tells about the behaviour of a vector with respect to
#   another, which is much broader in concept than mean value.
#
#   Therefore using correlation values or covariance values in the condition to classify, will
#   work better.
#   For example, we can use the below condition, where Corr() gives the correlation value
#     Corr(col1, col5) > (Corr(col1, col2) + Corr(col2, col3) + Corr(col1, col3))/3
#
#
#
#
