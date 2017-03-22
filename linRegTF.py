import tensorflow as tf
import numpy as np

# Create a y value which is approximately linear but with some random noise
train_X = np.linspace(-1, 1, 101)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")


def model(X, w):
	return tf.mul(X, w)

# Weight matrix
w = tf.Variable(0.0, name = "weights")
y_model = model(X, w)


cost = tf.square(Y - y_model)

# Build an optimizer to minimize the cost, fit line to the data
train_operation = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


with tf.Session() as sess:
	# Initialize variables e.g. W
	tf.global_variables_initializer().run()
	
	for i in range(100):
		for (x,y) in zip(train_X, train_Y):
			sess.run(train_operation, feed_dict={X:x, Y:y})
	print(sess.run(w))
