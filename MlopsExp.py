"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports

# Import datasets, classifiers and performance metrics
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
import pdb
from utils import preprocessing, split_train_dev_test, train_model, read_digits, predict_and_eval
##############################################################################
# #
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# 1. get the datasets
x, y = read_digits()


# 2. sanity check
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, x, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.



# Create a classifier: a support vector classifier

# 3. Data-Splitting
# Split data into 50% train and 50% test subsets

X_train, X_test, y_train, y_test = split_train_dev_test(x, y, test_size=0.3)


# 4. Data Preprocessing
# flatten the images
X_train = preprocessing(X_train)
X_test = preprocessing(X_test)


# 5. Model Training
# Learn the digits on the train subset
clf = train_model(X_train, y_train, {'gamma': 0.001})

# 6. Predict the value of the digit on the test subset
predict_and_eval(clf, X_test, y_test)