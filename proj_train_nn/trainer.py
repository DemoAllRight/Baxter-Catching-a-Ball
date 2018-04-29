import tensorflow as tf
import datetime
import os
import sys
import argparse

slim = tf.contrib.slim

# self.net.class_loss = loss
# self.train_step = optimizer
# self.train = train_op


class Solver(object):

    def __init__(self, net, data):

        # store the network
        self.net = net
        # store the data manager
        self.data = data
       
        #Number of iterations to train for
        self.max_iter = 5000
        #Every 200 iterations please record the trest and train loss
        self.summary_iter = 200
        

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        # self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train_step = tf.train.GradientDescentOptimizer(0.1)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        #Append the current train and test accuracy every 200 iterations
        self.train_loss = []
        self.test_loss = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        '''

        for ite in range(self.max_iter):
            print("iteration:", ite)
            train_features , train_labels = self.data.get_train_batch()

            _, c = self.sess.run([self.train, self.net.total_loss], feed_dict={self.net.features: train_features, self.net.labels: train_labels})

            loss = self.sess.run(self.net.class_loss, feed_dict={self.net.features: train_features, self.net.labels: train_labels})
            prediction = self.sess.run(self.net.logits, feed_dict={self.net.features: train_features[30:31,:], self.net.labels: train_labels[30:31,:]})
            self.train_loss.append(loss)
            print("training loss:", loss)
            print("prediction of training point:" , prediction)

            if(ite % self.summary_iter == 0 or ite == self.max_iter-1):
                test_features , test_labels = self.data.get_validation_batch()
                loss = self.sess.run(self.net.class_loss, feed_dict={self.net.features: test_features, self.net.labels: test_labels})
                self.test_loss.append(loss)
                print("test loss:", loss)
            else:
                self.test_loss.append(self.test_loss[-1])

            print('\n')

