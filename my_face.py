# Import required libraries
import tensorflow as tf              # TensorFlow for CNN model
import cv2                           # OpenCV for image processing and webcam
import dlib                          # dlib for face detection
import numpy as np                   # NumPy for numerical operations
import os                            # File and directory handling
import random                        # Random operations
import sys                           # System exit
from sklearn.model_selection import train_test_split  # Dataset splitting

# Paths to face datasets
my_faces_path = './my_faces'         # Images of authorized person
other_faces_path = './other_faces'   # Images of other people

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

# Function to read images, pad them, resize, and store labels
def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)

            # Add padding to make image square
            top, bottom, left, right = getPaddingSize(img)
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Resize image
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

# Load face datasets
readData(my_faces_path)
readData(other_faces_path)

# Convert data to NumPy arrays
imgs = np.array(imgs)

# Assign labels: [0,1] for my face, [1,0] for others
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(
    imgs, labs, test_size=0.05, random_state=random.randint(0, 100)
)

# Reshape image data
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# Normalize pixel values to [0,1]
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

# Batch size for training
batch_size = 128
num_batch = len(train_x) // batch_size

# Placeholders for TensorFlow inputs
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# Dropout probabilities
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# Initialize weights
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

# Initialize biases
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

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
    Wf = weightVariable([8 * 16 * 32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # Output layer
    Wout = weightVariable([512, 2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)

    return out

# Get model output
output = cnnLayer()

# Predicted class
predict = tf.argmax(output, 1)

# Restore trained model
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))

# Function to check if detected face is authorized
def is_my_face(image):
    res = sess.run(
        predict,
        feed_dict={x: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0}
    )
    return True if res[0] == 1 else False

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Open webcam
cam = cv2.VideoCapture(0)

# Real-time face recognition loop
while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)

    if not len(dets):
        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xff == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        # Crop and resize face
        face = img[x1:y1, x2:y2]
        face = cv2.resize(face, (size, size))

        print('Is this my face? %s' % is_my_face(face))

        # Draw rectangle around detected face
        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('image', img)

        if cv2.waitKey(30) & 0xff == 27:
            sys.exit(0)

# Close TensorFlow session
sess.close()
