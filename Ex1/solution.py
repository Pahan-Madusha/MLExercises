import numpy as np
import matplotlib.pyplot as plt
import unittest

####################
# Polynomial class #
####################
class Polynomial():
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def calculateVal(self, x_value):
        value = 0
        for i in range(0, len(self.coefficients)):
            value += (x_value**i)*self.coefficients[i]
        return value

    def differentiate(self, x_value):
        value = 0
        for i in range(1, len(self.coefficients)):
            value += (i*self.coefficients[i]*(x_value**(i-1)))
        return value

###################################
# Gradient descent algorithm class#
###################################
class GradientDescent():
    def __init__(self, precision, learn_rate, start, function):
        self.precision = precision
        self.learn_rate = learn_rate
        self.current_x = start
        self.step_size = precision+1
        self.function = function
        self.previous_x = self.current_x

    def run(self):
        while (self.step_size > self.precision):
            self.previous_x = self.current_x
            self.current_x = self.current_x - self.learn_rate*self.function.differentiate(self.current_x)
            self.step_size = abs(self.current_x - self.previous_x)
        return self.current_x


##########################
# Unit Tests (Polynomial)#
##########################
class PolynomialTest(unittest.TestCase):
    def setUp(self):
        self.function = Polynomial([1, 2, 3, 4])

    def testCalculateVal(self):
        self.assertEqual(self.function.calculateVal(0), 1)

    def testDifferentiate(self):
        self.assertEqual(self.function.differentiate(0), 2)

###############################
# Unit Tests (GradientDescent)#
###############################
class GradientDescentTest(unittest.TestCase):

    def setUp(self):
        self.function = Polynomial([1, 0, 3, -4])
        self.algorithm = GradientDescent(0.0000001, 0.001, -1, self.function)

    def testRun(self):
        self.assertAlmostEqual(self.algorithm.run(), 0, places=4, msg=None, delta=None)

# Run unit tests
if __name__ == '__main__':
    unittest.main(exit=False)

##############
# Initialize #
##############
function = Polynomial([3, 0, -3, -1]) # f(x) = -x^3 - 3x^2 + 3
precision = 0.00001
learn_rate = 0.001
start = -4
x_values = np.linspace(-3, 0, num=150)
y_values = [function.calculateVal(x) for x in x_values]

################
# Find Minimum #
################
algorithm = GradientDescent(precision, learn_rate, start, function)
min_ = algorithm.run()
print 'Local minimum at x = ' + str(min_)

########
# Plot #
########
plt.plot(x_values, y_values, 'b-')
plt.xlabel('$X$')
plt.ylabel('$f(x)$')
plt.title('$-x^3 - 3x^2 + 3$')
plt.axis([-3, -1, -1.2, 0])
plt.grid(True)
plt.text(-2.5, -1.1, 'Local minimum : x = ' + str(min_))
plt.show()

####################################################################################################
# (2) Initial x value was selected so that,
#		It was close to a local minima - Because in the algorithm, the current_x value will converge
#                                       quickly to the minimum with a closer initial value
#
#       It was not from part of the function that has very slowly increasing negative gradient - In my case,
#       for x>0, the gradient decreases very slowly(decrease rate decreases) and it can be presumed that gradient
#       becomes zero at infinity. Therefore in the program, integer overflow exception occur for Initial values > 0
#
#       It was not equal to any critical points of the function - In my case, there is a local maximum at x = 0.
#       If the initial value was selected as 0, the gradient at this point will be 0. Therefore the algorithm can't
#       proceed as the current_x value is stuck at the same value
#
# (3), (4)
#     If the learning rate is increased gradually, after some value, it will fail to produce the required precision.
#     Because the difference between current_x and previous_x will always be larger than the required precision.
#     Therefore the algorithm will run forever.
#
#     If the learning rate is kept within the above mentioned boundary, the smaller the learning rate, the larger will be
#     the time to find the minimum.
#
##############################################################################################################################