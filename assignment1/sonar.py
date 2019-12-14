################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################

import numpy as np
import pickle as pkl
import random

def perceptron(z):
    return -1 if z<=0 else 1

def ploss(yhat, y):
    return max(0, -yhat*y)

def ppredict(self, x):
    return self(x)

class Sonar_Model:

    def ppredict(self, x):
        return self(x)

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x), predict=ppredict):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):

        info = "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
        return info

    def __call__(self, x):
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat

    def load_model(self, file_path):
        with open(file_path, model="rb") as f:
            mm = pkl.load(f)
        self._dim = mm._dim
        self.w = mm.w
        self.b = mm.b
        self._a = mm._a


    def save_model(self):

        f = open("sonar_model.pkl", "wb")
        pkl.dump(self, f)
        f.close


class Sonar_Trainer:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def accuracy(self, data):
        acc = 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
        return acc

    def train(self, lr, ne):
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        for epoch in range(ne):
            for d in self.dataset:
                x = np.array(d[0])
                y = d[1]
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))


class Sonar_Data:

    def __init__(self, relative_path='../../data/assignment1/', data_file_name='sonar_data.pkl'):

        self.index = -1
        with open("%s%s" % (relative_path, data_file_name), mode = "rb") as f:
            sonar_data = pkl.load(f)
        self.samples = [(np.reshape(vector, vector.size), 1) for vector in sonar_data["m"]] + [(np.reshape(vector, vector.size), -1) for vector in sonar_data["r"]]
        random.shuffle(self.samples)

    def __iter__(self):
        return self

    def __next__(self):

        self.index += 1
        if self.index == len(self.samples):
            self.index = -1
            raise StopIteration
        return self.samples[self.index]

    def _shuffle(self):
        random.shuffle(self.samples)


def main():

    data = Sonar_Data()
    model = Sonar_Model(dimension=60, activation=perceptron, predict=ppredict)  # specify the necessary arguments
    trainer = Sonar_Trainer(data, model)
    trainer.train(0.01,1000) # experiment with learning rate and number of epochs
    model.save_model()

if __name__ == '__main__':

    main()
