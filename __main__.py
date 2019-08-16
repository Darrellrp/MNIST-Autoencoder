import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
import random

# Input
MNIST_IMG_DIM = [28, 28]

# Autocoder
HIDDEN_SIZE = 128
CODE_SIZE = 32

# Training
EPOCHS = 5
BATCH_SIZE = 32
VERBOSE = 1

# Optimization
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = None
DECAY = 0.0
AMSGRAD = False

# Post Training
EVALUATE = False
PREDICT = True
NUMBER_SHOWN_OF_PREDICTIONS = 4

# Load mnist dataset and generate train & test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_shape = x_train.shape[0]
INPUT_SIZE = np.prod(MNIST_IMG_DIM)

# Reshape datasets from (_, 28, 28) to (_, 784)
x_train = x_train.reshape(x_train.shape[0], INPUT_SIZE)
x_test = x_test.reshape(x_test.shape[0], INPUT_SIZE)

# Normalize RGB-values
x_train = np.true_divide(x_train, 255)
x_test = np.true_divide(x_test, 255)

# Convert to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

autoencoder = Sequential()

# Input layer (size=748) & Hidden layer 1
autoencoder.add(Dense(HIDDEN_SIZE, activation='relu', input_dim=INPUT_SIZE))
# Code layer
autoencoder.add(Dense(CODE_SIZE, activation='relu'))
# Hidden layer 2
autoencoder.add(Dense(HIDDEN_SIZE, activation='relu'))
# Output layer (Reconstructed image, siz=748)
autoencoder.add(Dense(INPUT_SIZE, activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

if EVALUATE:
	score, accuracy = autoencoder.evaluate(x_test, x_test)
	print('Test score:', score)
	print('Test accuracy:', accuracy)

if PREDICT:
	reconstructed = autoencoder.predict(x_test)

	# Reshape datasets from (_, 784) to (_, 28, 28)
	reconstructed = reconstructed.reshape(x_test.shape[0], MNIST_IMG_DIM[0], MNIST_IMG_DIM[1])
	actual = x_test.reshape(x_test.shape[0], MNIST_IMG_DIM[0], MNIST_IMG_DIM[1])

	# Show a number of predictions
	for i in range(NUMBER_SHOWN_OF_PREDICTIONS):
		# Generate random index
		rand_i = random.randrange(0, x_test.shape[0])

		print('')

		# Print actual label
		print('The label is %d' % y_test[rand_i])

		# Show actual image
		plt.imshow(actual[rand_i], cmap='Greys')
		plt.show()

		# Show reconstructed image
		plt.imshow(reconstructed[rand_i], cmap='Greys')
		plt.show()
