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

def sigmoid(z):
    return

def lrloss(yhat, y):
    return

def lrpredict(self, x):
    return

class Cat_Model:

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=None):

        pass

    def __str__(self):
        '''
        display the model's information
        '''
        info = ""
        return info

    def __call__(self, x):
        '''
        return the output of the network
        '''
        yhat = None
        return yhat

    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        pass

    def save_model(self):
        '''
        save your model as 'cat_model.pkl' in the local path
        '''
        pass

class Cat_Trainer:

    def __init__(self, dataset, model):

        pass

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        acc = None
        return acc

    def train(self, lr, ne):
        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''

class Cat_Data:

    def __init__(self, relative_path='../../data/assignment1/', data_file_name='cat_data.pkl'):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        pass

    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''
        return self

    def __next__(self):
        '''
        See example code (ngram) in slides
        '''
        return

    def _shuffle(self):
        '''
        shuffle the data iterator
        '''
        pass

def main():

    data = Sonar_Data()
    model = Sonar_Model()  # specify the necessary arguments
    trainer = Sonar_Trainer(data, model)
    trainer.train() # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()
