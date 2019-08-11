#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:54:08 2019

@author: xjchen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

from flask import Flask
from flask import Flask, render_template,request
from flask import jsonify


import argparse
import sys

import tensorflow as tf

FLAGS = None

def main(_):
        #import data 
        
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    
    


    #Defining Tensorflow variables, #W takes the format of [i,j].j representing a
    # given image x pixel index
        
    x = tf.compat.v1.placeholder("float",[None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
        
    #parameter settings for exponential decay for learning rate
        
    learning_rate_base = 0.9
    learning_rate_decay = 0.99
    batch_size = 100
    batch_num = mnist.train.num_examples // batch_size
        
    global_step = tf.Variable(0,trainable = False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                                   batch_num, learning_rate_decay, 
                                                   staircase = False)
    
        
    #cross-entropy
        
    y_ = tf.placeholder("float",[None,10])
    cross_entropy = -tf.reduce_sum(y_*tf.math.log(y))
    
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(
            cross_entropy, global_step = global_step)
        
        
    #Initialize the variables
        
    init = tf.initialize_all_variables()
    

    #model evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    saver = tf.train.Saver()
    
    sess = tf.compat.v1.Session()
    sess.run(init)    
    
    for epoch in range(1001):
        #regression 21 time
        if epoch % 10 == 0:
            save_path = saver.save(sess,"./my_model.ckpt")
            
            for batch in range(batch_num):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.test.labels})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        
        print ("After" + str(epoch) + "training steps," + "training accuracy is:" +str(train_acc)
        +", testing accuracy is: " + str(test_acc))
    
    save_path = saver.save(sess, "./my_model_final.ckpt")
        
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='home/xjchen/Documents/python/docker-compose/app/input_data',
                                 help = 'Directory for storing input data')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
    
'''
parameter names:
    training dataset - mnist.train.images
    training dataset label - mnist.train.labels
    
28x28
grey scale 0-1

[60000,784]tensor. This syntax enables reference to any pixel of any of the 
images

one-hot vectors is used in the code to represent the dataset tag in the format
of [60000,10]

y=softmax(evidence)

softmax(x) = normalize (exp(x))
'''
