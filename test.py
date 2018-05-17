##Loading dataset and build model as done in train.py

# Import necessary modules
import tensorflow as tf
import cv2
import glob
import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
# from matplotlib import pyplot

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("dir_path", help="Provide data path of the directory containing road_bikes and mountain_bikes folders ")
# args = parser.parse_args()

# Load dataset
# Read Dataset
road_bikes = glob.glob("/home/sarala/Downloads/Dataset_bikes/road_bikes/*.jpg")
mountain_bikes = glob.glob("/home/sarala/Downloads/Dataset_bikes/mountain_bikes/*.jpg")

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
                                                                                    labels, test_size=0.3,
                                                                                    shuffle=True, random_state=42)
# Scale data

image_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training and testing
I_train = image_scaler.fit_transform(I_train)
# It's very important that the training and test data are scaled with the same scaler.
I_test = image_scaler.transform(I_test)

# Define model parameters
learning_rate = 0.001

# Define how many inputs and outputs are in our neural network
number_of_inputs = 200*100
number_of_outputs = 2

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 32
layer_2_nodes = 64
layer_3_nodes = 1024
drop_layer_keep_rate = 0.8 # layer 4 drop out

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Reduce image size by factor 2 using maxpool
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=([None, 100, 200, 1]))

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
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    saver.restore(session, 'logs/trained_model.ckpt')
    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    # prediction = tf.nn.log_softmax(cost)
    I_test = np.reshape(I_test, (-1, 100, 200, 1))
    predicted_label = session.run(tf.nn.softmax(prediction), feed_dict={X: I_test})

    # Compare the prediction and the labels, Accuracy = mean(number of correct prediction)
    correct = tf.equal(tf.argmax(predicted_label, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Testing accuracy', accuracy.eval({X: I_test, Y: label_test}))


print('Predicted labels = {}'.format(predicted_label))
print('labels',label_test)

# I_test = np.reshape(I_test,(100,200))
# fig = plt.figure(figsize=(25, 4))
# for i in np.arange(len(label_test)):
#     ax = fig.add_subplot(2, len(label_test)/2, i+1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(I_test[i]), cmap='gray')
#     ax.set_title("{} ({})".format(predicted_label[i,:], label_test[i,:]),
#                  color=("green" if predicted_label[i]==label_test[i] else "red"))



