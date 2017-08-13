import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.util.testing import assert_frame_equal

class AnglePredictor:
    def __init__(self, channels_fName, angle_fName):
        self.channel_fName = channels_fName
        self.angle_fName = angle_fName
        self.channel_index = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5']
        self.angle_index = ['angle1', 'angle2', 'angle3']

    def readData(self):
        self.channels = pd.read_csv(self.channel_fName, names = self.channel_index)
        self.angles = pd.read_csv(self.angle_fName, names = self.angle_index)

    def mergeData(self):
        self.data = self.channels.join(self.angles)

    def logReg(self, X, _y):
        y = _y.astype(int).values.ravel() # Convert float angle values to int to use logistic regression
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_proba = log_reg.predict(X_val)
        accuracy = log_reg.score(X_val, y_val)
        return y_proba, X_val, y_val, accuracy

    def linearReg(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001)
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_proba = lin_reg.predict(X_val)
        return y_proba

    def getAccuracy(self, x, y):
        lin_reg = LinearRegression()
        return lin_reg.score(x, y)
###########################################################
######################### Testing #########################
###########################################################
class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.dataFrame1 = pd.DataFrame({
            'a' : [1, 2, 3]
        })
        self.dataFrame2 = pd.DataFrame({
            'b': [4, 5, 6]
        })
        self.ap = AnglePredictor('', '')
        self.ap.channels = self.dataFrame1
        self.ap.angles = self.dataFrame2

    def testMergeData(self):
        self.ap.mergeData()
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        assert_frame_equal(expected, self.ap.data)


# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)

ap = AnglePredictor('ExerciseChannels.csv', 'ExerciseAngles.csv')
ap.readData() # (1)
ap.mergeData() # (2)

# Observe correlation matrix to select suitable data columns
print 'Correlation matrix'
print ap.data.corr()
print ''

print 'Predicted results'
predictions, sample_data, labels, accuracy = ap.logReg(ap.data[['channel2', 'channel5']], ap.data[['angle2']])
print predictions
print ''

print 'Expected values'
print labels
print ''

print 'Accuracy score'
print accuracy

########################################################
# 3) Logistic regression was selected for model building. Because the angle values range from
#    0-180 which is a comparatively a small range of values. The predicting attribute(angle) can
#    therefore be categorized in to one of integer values from 0 to 180. Since the given dataset
#    considerably large this is possible.
#
# 4)          channel1  channel2  channel3  channel4  channel5    angle1  \
#   channel1  1.000000  0.830008  0.796636  0.793410  0.417799 -0.068256
#   channel2  0.830008  1.000000  0.858349  0.840664  0.563501 -0.020387
#   channel3  0.796636  0.858349  1.000000  0.903540  0.608883 -0.005799
#   channel4  0.793410  0.840664  0.903540  1.000000  0.727946  0.042314
#   channel5  0.417799  0.563501  0.608883  0.727946  1.000000  0.154638
#   angle1   -0.068256 -0.020387 -0.005799  0.042314  0.154638  1.000000
#   angle2   -0.056810 -0.022127 -0.006331  0.036965  0.114981  0.884187
#   angle3    0.087207  0.033709  0.002751 -0.030646 -0.161679 -0.834815
#
#              angle2    angle3
#   channel1 -0.056810  0.087207
#   channel2 -0.022127  0.033709
#   channel3 -0.006331  0.002751
#   channel4  0.036965 -0.030646
#   channel5  0.114981 -0.161679
#   angle1    0.884187 -0.834815
#   angle2    1.000000 -0.859589
#   angle3   -0.859589  1.000000
#
#   This is the correlation matrix of the data set.
#
#   # Selecting channels #
#   When correlation between channels are observed, some of the channels are highly
#   correlated to each other. Therefore we can omit one channel from a pair
#   of highly correlated channels
#
#   When we do this, we are left with channel2 and channel5
#   channel2 and channel5 are not very correlated with each other but they are highly
#   correlated with other channels. Therefore other channels can be ignored as channel2
#   and channel5 can represent the data set well.
#   (Although channel1 and channel5 have a lower correlation value, their correlation with
#    other channels is not very high to ignore)
#
#   # Selecting angle #
#   angle2 is selected as the label attribute data.
#   When we consider the consider the correlation values between
#       anglex and channel2 (corr1)
#       anglex and channel5 (corr2)
#   The gap between these two values (|corr1 - corr2|) is the lowest for angle2
#   when compared with that of angle1 and angle3. Since closely correlated data with
#   labeling attribute is preferred in this case, angle2 was selected.
#
# (5) The prediction accuracy is rather low in this model. The reason can be that the dataset is
#     biased with similar values which cause the model to overfit across some values. Therefore
#     there is a bias for some test data in the prediction values. For example in the angle2 values
#     168 is very commonly seen.
#



