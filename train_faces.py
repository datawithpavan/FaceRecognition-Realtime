# Import required libraries
import tensorflow as tf                    # TensorFlow for CNN model
import cv2                                 # OpenCV for image processing
import numpy as np                         # NumPy for numerical operations
import os                                  # File handling
import random                              # Random utilities
import sys                                 # System exit
from sklearn.model_selection import train_test_split  # Train-test split

# Dataset paths
my_faces_path = './my_faces'               # Positive samples
other_faces_path = './other_faces'         # Negative samples

# Image size
size = 64

# Lists to store images and labels
imgs = []
labs = []

# Function to calculate padding so image becomes square
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top

    return top, bottom, left, right

# Function to read images, pad, resize, and label them
def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            # Read image
            img = cv2.imread(filename)

            # Pad image to make it square
            top, bottom, left, right = getPaddingSize(img)
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Resize image
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

# Load datasets
readData(my_faces_path)
readData(other_faces_path)

# Convert lists to NumPy arrays
imgs = np.array(imgs)

# One-hot encode labels
# [0,1] → my face, [1,0] → other faces
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# Split dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(
    imgs, labs, test_size=0.05, random_state=random.randint(0, 100)
)

# Reshape data to match CNN input format
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# Normalize pixel values to [0,1]
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

# Batch size
batch_size = 100
num_batch = len(train_x) // batch_size

# TensorFlow placeholders
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# Dropout probabilities
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# Initialize weights
def weightVariable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Initialize biases
def biasVariable(shape):
    return tf.Variable(tf.random_normal(shape))

# Convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pooling operation
def maxPool(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )

# Dropout operation
def dropout(x, keep):
    return tf.nn.dropout(x, keep)

# CNN architecture definition
def cnnLayer():
    # First convolution layer
    W1 = weightVariable([3, 3, 3, 32])
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    # Second convolution layer
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # Third convolution layer
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # Fully connected layer
    Wf = weightVariable([8 * 8 * 64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # Output layer
    Wout = weightVariable([512, 2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)

    return out

# CNN training function
def cnnTrain():
    out = cnnLayer()

    # Cross-entropy loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_)
    )

    # Adam optimizer
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    # Accuracy calculation
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32)
    )

    # TensorBoard summaries
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    # Model saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        # Train for multiple epochs
        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size:(i + 1) * batch_size]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size]

                _, loss, summary = sess.run(
                    [train_step, cross_entropy, merged_summary_op],
                    feed_dict={
                        x: batch_x,
                        y_: batch_y,
                        keep_prob_5: 0.5,
                        keep_prob_75: 0.75
                    }
                )

                summary_writer.add_summary(summary, n * num_batch + i)
                print(n * num_batch + i, loss)

                # Evaluate model every 100 steps
                if (n * num_batch + i) % 100 == 0:
                    acc = accuracy.eval({
                        x: test_x,
                        y_: test_y,
                        keep_prob_5: 1.0,
                        keep_prob_75: 1.0
                    })
                    print(n * num_batch + i, acc)

                    # Save model if accuracy is high enough
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n * num_batch + i)
                        sys.exit(0)

        print('Accuracy less than 0.98, training stopped.')

# Start CNN training
cnnTrain()
