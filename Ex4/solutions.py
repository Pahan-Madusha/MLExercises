import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal, assert_frame_equal
from sklearn import tree
import os
import pydotplus
from IPython.display import Image
from PIL import Image as Img
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, fName):
        self.fName = fName
        self.classifier = tree.DecisionTreeClassifier()

    # Read data
    def readData(self):
        self.data = pd.read_csv(self.fName)

    # Handle missing values
    def handleMissingValues(self):
        if hasattr(self, 'data'):
            #self.data.dropna(axis=0, inplace=True, thresh=8) # Drop rows with more than 7 missing values
            #self.data.reset_index(drop=True, inplace=True)
            self.data.fillna(self.data.mean(), inplace=True) # Fill remaining missing values with column mean

    # Convert a column to binary data
    def columnToBinary(self, columnName, threshold):
        self.data[columnName] = np.where(self.data[columnName] > threshold, 1, 0)

    # Split data
    def splitData(self):
        train, self.test_data = train_test_split(self.data, test_size=0.3333)
        self.training_data = train.drop(['COUNTRY', 'BREASTCANCERPER100TH'], axis=1)  # Remove country and targets
        self.targets = train['BREASTCANCERPER100TH']

    # Initialize the classifier
    def fit(self):
        self.classifier = self.classifier.fit(self.training_data, self.targets)

    # Generate graph
    def gen_graph(self):
        with open("bcancer.dot", 'w') as f:
            f = tree.export_graphviz(self.classifier, out_file=f)
        os.unlink('bcancer.dot')
        dot_data = tree.export_graphviz(self.classifier, out_file=None)
        self.graph = pydotplus.graph_from_dot_data(dot_data)

    # Add colors
    def gen_colored_graph(self):
        feature_names = list(self.data.columns.values)
        feature_names.remove('COUNTRY')
        feature_names.remove('BREASTCANCERPER100TH')
        dot_data = tree.export_graphviz(self.classifier, out_file=None, feature_names=feature_names,class_names=['A', 'B'], filled=True, rounded=True, special_characters=True)
        self.graph = pydotplus.graph_from_dot_data(dot_data)

    # Write decision tree to a pdf
    def write_to_pdf(self, fName):
        self.graph.write_pdf(fName)

    # Write decision tree to a png
    def write_to_png(self, fName):
        Image(self.graph.create_png())
        self.graph.write_png(fName)
        image = Img.open(fName)
        image.show()

    # Predict using model
    def predict(self):
        test = self.test_data.drop(['COUNTRY', 'BREASTCANCERPER100TH'], axis=1)
        predicted = self.classifier.predict(test)
        actual = (self.test_data['BREASTCANCERPER100TH']).as_matrix()
        accuracy = accuracy_score(actual, predicted) * 100
        countries = self.test_data['COUNTRY']
        countries.reset_index(drop=True, inplace=True)
        return countries, predicted, accuracy

###########################################################
######################### Unit Tests #########################
###########################################################
class DecisionTreeTest(unittest.TestCase):
    def setUp(self):
        self.dt = DecisionTree('')
        self.dt.data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2.3, np.NaN, 3.5, 1.3, 4.9]
        })

    def testColumnToBinary(self):
        expected = pd.Series([0, 0, 0, 1 , 1], name='a')
        self.dt.columnToBinary('a', 3)
        assert_series_equal(expected, self.dt.data['a'])

    def testHandleMissingValues(self):
        expected = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2.3, 3, 3.5, 1.3, 4.9]
        })
        self.dt.handleMissingValues()
        assert_frame_equal(expected, self.dt.data)

# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)

#########################################################
################## End of tests #########################
#########################################################

dt = DecisionTree("breaset-cancer.csv")
dt.readData() # Question (1)
dt.handleMissingValues() # Question (1)
dt.columnToBinary('BREASTCANCERPER100TH', 20) # Question (2)

# # Question (3), (4)
dt.splitData()
dt.fit()
dt.gen_colored_graph()
dt.write_to_png("tree.png")
countries, prediction, accuracy = dt.predict()
results = countries.to_frame().assign(prediction=pd.DataFrame(prediction))

print results.to_string(index=False)
print ''
print 'Accuracy : ',accuracy,'%'  # Question (5)





