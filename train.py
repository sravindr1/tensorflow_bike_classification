# Import necessary modules
import tensorflow as tf
import cv2
import glob
import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("dir_path", help="Provide data path of the directory containing road_bikes and mountain_bikes folders ")
# args = parser.parse_args()

# Load dataset
# Read Dataset
road_bikes = glob.glob("/home/sarala/PycharmProjects/P1/bikes/road_bikes/*.jpg")
mountain_bikes = glob.glob("/home/sarala/PycharmProjects/P1/bikes/mountain_bikes/*.jpg")

# road_bikes = glob.glob(args.dir_path +"/road_bikes/*.jpg")
# mountain_bikes = glob.glob(args.dir_path + "/mountain_bikes/*.jpg")

print('Number of road bike images = ', len(road_bikes))
print('Number of road bike images = ', len(mountain_bikes))


# To read images (gray scale and resized to (200,100)) ,
# The size was chosen to be able to run on my poor system large size images threw memory allocation errors
def load_img_path(img_dir):
    img = []
    for t in (img_dir):
        im = cv2.resize(cv2.imread(t, 0), (200, 100)).flatten()
        img.extend(im)
    img = np.array(img).reshape((len(img_dir), 100 * 200))
    return img


# Label gereration
labels = []
for i in range(len(road_bikes)):
    labels.extend((1, 0))
for i in range(len(mountain_bikes)):
    labels.extend((0, 1))
labels = np.array(labels).reshape(len(road_bikes) + len(mountain_bikes), -1)
print('Done labeling,road bike label = (1,0) mountain bike label  = (0,1)')
print('labels shape = ', labels.shape)

# Create train and test Dataset
road_dataset = load_img_path(road_bikes)
mountain_dataset = load_img_path(mountain_bikes)
dataset = np.vstack((road_dataset, mountain_dataset))
print('finished dataset loading, dataset shape = ', dataset.shape)
print('Splitting dataset to training and testing dataset')
# Split train and test data
I_train, I_test, label_train, label_test = sklearn.model_selection.train_test_split(dataset,
                                                                                    labels, test_size=0.15,
                                                                                    shuffle=True, random_state=42)
# Scale data
image_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training and testing
I_train = image_scaler.fit_transform(I_train)
# It's very important that the training and test data are scaled with the same scaler.
I_test = image_scaler.transform(I_test)

# Build model
# Define model parameters
learning_rate = 0.001
training_epochs = 75
batch_size = 20
print_every_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 200*100
number_of_outputs = 2

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 32
layer_2_nodes = 64
layer_3_nodes = 1024
drop_layer_keep_rate = 0.8 # layer 4 drop out

# Section One: Define the layers of the neural network itself

# Define functions to be used in convolutional neural network
# convolutional layer stride =[1,1,1,1] and output size same as input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Reduce image size by factor 2 using maxpool
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=([None, 100,200,1]))

# Layer 1 conv2d-->relu-->maxpool
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape= ([3, 3, 1, layer_1_nodes]), initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = maxpool2d(tf.nn.relu(conv2d(X, weights) + biases))

# Layer 2 1 conv2d-->relu-->maxpool
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=([3, 3, layer_1_nodes, layer_2_nodes]), initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = maxpool2d(tf.nn.relu(conv2d(layer_1_output, weights) + biases))

# Layer 3 fully connected layer
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[25*50*layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.reshape(layer_2_output, [-1, 25 * 50 * layer_2_nodes])
    layer_3_output = tf.add(tf.matmul(layer_2_output, weights), biases)

# Layer 4 Drop out layer
with tf.variable_scope('layer_4'):
    layer_4_output = tf.nn.dropout(layer_3_output, keep_prob=drop_layer_keep_rate)

# Layer 5 Output
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights5", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases5", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_4_output, weights) + biases

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 2))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# To save model graph
saver = tf.train.Saver()
# writer = tf.Summary.graph()
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Write graph in graphs file in the working directory
    writer = tf.summary.FileWriter('/home/sarala/PycharmProjects/P1/graphs', session.graph)

    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):
        # Feed in the training data and do one epoch of neural network training
        epoch_loss = 0

        # Batch training
        # Index to keep track of input
        i = 0
        while i < len(I_train):
            start = i
            end = i + batch_size

            # Define batch input and corresponding batch labels to train the model
            # Run the cost and optimizer function on graph, keeping track of loss for every batch of training,
            # update input index
            batch_x = np.array(I_train[start:end])
            batch_x = np.reshape(batch_x, (-1, 100, 200, 1))
            batch_y = np.array(label_train[start:end])
            _, training_cost = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            epoch_loss += training_cost
            i += batch_size

        # Print the current training status to the screen
        if epoch % print_every_step == 0:

            print('Epoch = {}, Training loss = {} '.format(epoch+1, epoch_loss))



    # Training is now complete!
    print("Training complete!")

    # Compare the prediction and the labels, Accuracy = mean(number of correct prediction)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    I_train = np.reshape(I_train,(-1,100,200,1))
    print('Training accuracy', accuracy.eval({X: I_train, Y: label_train}))

    # Save model
    save_path = saver.save(session, 'logs/trained_model.ckpt')
    print('Model saved {}'.format(save_path))
