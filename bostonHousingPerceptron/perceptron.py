import numpy as np
import pandas as pd
import random as rand

"""
for this assignment we were given a little bit of a start but it was pretty much just function def's and comments on 
what we may want to do in it
"""
class Perceptron(object):

    def __init__(self):
        """
        Create and initialize a new Perceptron object.
        """
        d = {"crime": [1], "zn": [1], "indus": [1], "chas": [1], "nox": [1], "rm": [1], "age": [1], "dis": [1],
             "rad": [1], "tax": [1], "ptratio": [1], "b": [1], "lstat": [1], "bias": [1]}
        self.weights = pd.DataFrame(d).as_matrix()
        pass

    def predict(self, x):
        """Predict the class of sample x. (Forward pass)"""
        weighted_vector = np.multiply(x, self.weights)
        predictor = 0
        for i in range(len(weighted_vector[0])-1):
            predictor += weighted_vector[0][i]
        if predictor > 0:
            return 1
        else:
            return 0

    def _delta(self, x, y_hat, y):
        """
        Given predictions y_hat and targets y, calculate the weight
        update delta.
        """
        d = np.multiply((y - y_hat), x)
        return d

    def _update_weights(self, delta):
        """
        Update the weights by delta.
        """
        self.weights = np.add(self.weights, delta)
        pass

    def _train_step(self, x, y):
        """
        Perform one training step:
            - predict
            - calculate delta
            - update weights
        Returns the predictions y_hat.
        """
        y_hat = self.predict(x)
        delta = self._delta(x, y_hat, y)
        self._update_weights(delta)
        #print(y - y_hat)
        return y_hat

    def train(self,
              train_x,
              train_y,
              test_x,
              test_y,
              num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        """
        temp_a = 0
        temp_w = self.weights
        for x in range(num_steps):
            for i in range(len(train_x)):
                self._train_step(train_x[i], train_y[i])
            # check to see if it is more accurate now
            accuracy = test(self, test_x, test_y)
            if temp_a < accuracy:
                temp_a = accuracy
                temp_w = self.weights

        self.weights = temp_w
        pass


def scale(data):
    return data / data.max()


def test(p, x_test, y_test):
    # loop through each of the testing data and see how well it does
    right = 0
    for i in range(len(x_test)):
        y = p.predict(x_test[i])
        res = y - y_test[i]
        if res == 0:
            right += 1
    return right / len(x_test)


def main():
    # read data into a dataframe
    data = pd.read_csv("./bostonHousingD.csv")
    sdata = scale(data)
    # get data frame ready for in format to learn from
    smedv_mean = sdata.medv.mean()
    sdata.medv = pd.Series(np.where(sdata.medv.values < smedv_mean, 1, 0), sdata.index)
    ones = np.ones((506, 1))
    ones = pd.DataFrame(ones)
    sdata = pd.concat([sdata, ones], axis=1)
    sdata = sdata.rename(index=str, columns={0: "bias"})
    # split data into a learner set and a tester set
    learner = sdata.iloc[0:1]
    tester = sdata.iloc[1:2]
    i = 2
    while i < 506:
        if i % 10 == rand.randint(0, 10):  # can do this or use an static number here
            tester = tester.append(sdata.iloc[i:i+1], ignore_index=True)
        else:
            learner = learner.append(sdata.iloc[i:i+1], ignore_index=True)
        i = i + 1
    # print(learner.head())
    # print(tester.head())
    p = Perceptron()
    # get the learner set ready to train the perceptron
    medvVec = learner[['medv']]
    medvVec = medvVec.as_matrix()
    learner = learner.drop(columns=['medv'])
    learner = learner.as_matrix()
    # get the tester set ready to test the perceptron
    answer = tester[['medv']]
    answer = answer.as_matrix()
    tester = tester.drop(columns=['medv'])
    tester = tester.as_matrix()
    # train and test
    p.train(learner, medvVec, tester, answer, 2000)
    accuracy = test(p, tester, answer)
    print(accuracy)  # check the accuracy


if __name__ == '__main__':
    main()
