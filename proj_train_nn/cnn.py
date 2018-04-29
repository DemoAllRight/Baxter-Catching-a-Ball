
import numpy as np
import tensorflow as tf
#import yolo.config_card as cfg

# import IPython

slim = tf.contrib.slim


class CNN(object):

    def __init__(self,output_size,feature_size):
        '''
        Initializes the size of the network
        '''

        # define the size of input and output
        self.output_size = output_size
        self.feature_size = feature_size
        self.batch_size = 40

        # configure and build the network
        self.features = tf.placeholder(tf.float32, [None, self.feature_size], name='features')
        self.labels = tf.placeholder(tf.float32, [None, self.output_size])
        self.logits = self.build_network(self.features, num_outputs=self.output_size)

        # define the loss
        # self.loss_layer(self.logits, self.labels)
        # self.loss_layer(self.logits, self.labels)
        self.class_loss = tf.sqrt(tf.reduce_mean(tf.pow(self.logits-self.labels,2)))

        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      features,
                      num_outputs,
                      scope='yolo'):

        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.000005)):

                '''
                Fill in network architecutre here
                Network should start out with the images function
                Then it should return net
                '''
                # net = slim.flatten(net, scope='flat')
                net = slim.fully_connected(features, 1000, scope='fc_0')
                net = slim.fully_connected(net, 1000, scope='fc_1')
                net = slim.fully_connected(net, 1000, scope='fc_2')
                net = slim.fully_connected(net, 1000, scope='fc_3')
                net = slim.fully_connected(net, 1000, scope='fc_4')
                # net = slim.fully_connected(net, 1000, scope='fc_5')
                net = slim.fully_connected(net, num_outputs, scope='fc_5')

        return net


    # def get_acc(self,y_,y_out):

    #     '''
    #     compute accurracy given two tensorflows arrays
    #     y_ (the true label) and y_out (the predict label)
    #     '''

    #     cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))

    #     ac = tf.reduce_mean(tf.cast(cp, tf.float32))

    #     return ac

    # def loss_layer(self, predicts, labels, scope='loss_layer'):
    #     '''
    #     The loss layer of the network, which is written for you.
    #     You need to fill in get_accuracy to report the performance
    #     '''
    #     with tf.variable_scope(scope):
    #         t = predicts - labels
    #         self.class_loss = tf.reduce_mean(tf.nn.l2_loss(t))

    #         # self.accurracy = self.get_acc(classes,predicts)
