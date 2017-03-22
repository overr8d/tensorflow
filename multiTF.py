#!/usr/bin/env python3

import tensorflow as tf

# Create variables
a = tf.placeholder("float")
b = tf.placeholder("float")

# Multiplication
y = tf.mul(a,b) 

with tf.Session() as sess: 
	print("%f should equal 4.0" % sess.run(y, feed_dict={a:2, b:2}))
	print("%f should equal 12.0" % sess.run(y,feed_dict={a:4, b:3}))
