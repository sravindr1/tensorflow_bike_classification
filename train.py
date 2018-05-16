# Import necessary modules
import tensorflow as tf
import cv2
import glob
import numpy as np
import sklearn.model_selection

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dir_path", help="Provide data path of the directory containing road_bikes and mountain_bikes folders ")
args = parser.parse_args()

# Read Dataset
# road_bikes = glob.glob("/home/sarala/Downloads/Dataset_bikes/road_bikes/*.jpg")
# mountain_bikes = glob.glob("/home/sarala/Downloads/Dataset_bikes/mountain_bikes/*.jpg")

road_bikes = glob.glob(args.dir_path +"/road_bikes/*.jpg")
mountain_bikes = glob.glob(args.dir_path + "/mountain_bikes/*.jpg")

print('Number of road bike images = ', len(road_bikes))
print('Number of road bike images = ', len(mountain_bikes))


# To read images (gray scale and resized to (200,100)) ,
# The size was chosen to be able to run on my poor system large size images threw memory allocation errors
def load_img_path(img_dir):
    img = []
    for t in (img_dir):
        im = cv2.resize(cv2.imread(t, 0), (200, 100)).flatten()
        img.extend(im)
    img = np.array(img).reshape((len(img_dir), 100 * 200)) / 255.0
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
# n_classes = number of output class,
# batch_size = batch size in used in training
n_classes = 2
batch_size = 10

# Placeholder for input image data x and output labels
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 100 * 200])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)


# Define functions to be used in convolutional neural network
# convolutional layer stride =[1,1,1,1] and output size same as input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Reduce image size by factor 2 using maxpool
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# Define convolutional neural network
def conv_net(x):
    # Define weights and biases for layers
    with tf.name_scope('weights'):
        wieghts = {  # 3*3 convolution, 1 input image, 32 outputs
                    'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
                    # 3*3 conv, 32 inputs, 64 outputs
                    'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
                    # fully connected, 25*50*64 inputs, 1024 outputs
                    'W_fc2': tf.Variable(tf.random_normal([25 * 50 * 64, 1024])),
                    # 1024 inputs, 2 outputs (class prediction)
                    'W_out': tf.Variable(tf.random_normal([1024, n_classes]))}

    with tf.name_scope('biases'):
        biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                      'b_conv2': tf.Variable(tf.random_normal([64])),
                      # 'b_fc1': tf.Variable(tf.random_normal([10240])),
                      'b_fc2': tf.Variable(tf.random_normal([1024])),
                      'b_out': tf.Variable(tf.random_normal([n_classes]))}

    # Define CNN architecture
    # Reshape input to tensor (1 gray channel)
    with tf.name_scope('input_reshape'):
        x = tf.reshape(x, shape=[-1, 100, 200, 1])

    # Two [ convolution --> relu --> maxpool ] layers
    with tf.name_scope('Conv1_Relu_Maxpool1'):
        conv1 = maxpool2d(tf.nn.relu(conv2d(x, wieghts['W_conv1']) + biases['b_conv1']))
    with tf.name_scope('Conv2_Relu_MaxPool2'):
        conv2 = maxpool2d(tf.nn.relu(conv2d(conv1, wieghts['W_conv2']) + biases['b_conv2']))

    # Two fully connected layers
    with tf.name_scope('Fcn1_layer'):
        fc1 = tf.reshape(conv2, [-1, 25 * 50 * 64])
    # fc1 = tf.nn.relu(tf.matmul(fc1, wieghts['W_fc1']) + biases['b_fc1'])
        fc2 = tf.nn.relu(tf.matmul(fc1, wieghts['W_fc2']) + biases['b_fc2'])

    with tf.name_scope('Drop_out_layer'):
        fc2 = tf.nn.dropout(fc2,keep_prob=0.8)

    with tf.name_scope('Ouptut_layer'):
        output = tf.add(tf.matmul(fc2, wieghts['W_out']), biases['b_out'])

    return (output)

# Train model
def train_nn(x):

    # Predict labels on train set ,Calculate the mean cross-entropy loss(prediction,labels) = cost
    # Optimise minimum cost, using Adam optimizer, learning rate = 0.001 default
    with tf.name_scope('Training'):
        with tf.name_scope('Prediction_logits'):
            prediction = conv_net(x)
        with tf.name_scope('Optimize_cost'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        # number of epochs
        epochs = 5
        # Object to save model
        saver = tf.train.Saver()
        # train_nn includes session run to compute loss and optimize min cost with input in batches of batch_size
        with tf.Session() as sess:
            # Initialize global variables within session
            sess.run(tf.global_variables_initializer())
            # Write graph in graphs file in the working directory
            writer = tf.summary.FileWriter('./graphs', sess.graph)

            # Train model epochs number of times
            for epoch in range(epochs):
                epoch_loss = 0
                # Index to keep track of input

                i = 0
                while i < len(I_train):
                    start = i
                    end = i + batch_size

                    # Define batch input and corresponding batch labels to train the model
                    # Run the cost and optimizer function on graph, keeping track of loss for every batch of training,
                    # update input index
                    batch_x = np.array(I_train[start:end])
                    batch_y = np.array(label_train[start:end])
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    i += batch_size

                # Print loss every epoch
                print('epoch', epoch+1, 'out of', epochs, 'loss', epoch_loss)

            # Compare the prediction and the labels, Accuracy = mean(number of correct prediction)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Training accuracy', accuracy.eval({x: I_train, y: label_train}))
            # Save model
            saver.save(sess, '/home/sarala/PycharmProjects/P1/C/train_model/save_net.ckpt')

            # Testing part as i could not successfully restore the model onto another file I write the test code
            # in the same file train.py
            print('test scores',sess.run(tf.nn.softmax(prediction), feed_dict={x: I_test}))
            print('test label', label_test)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Testing accuracy', accuracy.eval({x: I_test, y: label_test}))

        writer.close()

train_nn(x)
