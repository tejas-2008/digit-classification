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
import itertools
from utils import *
import numpy as np
import os
classifier_param_dict = {}
test_size_array = [0.2]
dev_size_array = [0.2]

# SVM
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
h_param_svm = {'gamma': gamma_ranges, 'C': C_ranges}
h_param_svm_comb = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_ranges, C_ranges)]

classifier_param_dict["svm"] = h_param_svm_comb

# Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_param_tree = {}
h_param_tree['max_depth'] = max_depth_list
h_param_tree_comb = [{'max_depth': max_depth} for max_depth in max_depth_list]
classifier_param_dict["DecisionTree"] = h_param_tree_comb


solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
h_param_lr = {}
h_param_lr['solver'] = solver
h_param_lr_comb = [{'solver': sol} for sol in solver]
classifier_param_dict["logisticRegression"] = h_param_lr_comb
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
# Split data into test, train and dev data

def run_exp():
    for test_size in test_size_array:
        for dev_size in dev_size_array:
            X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(x, y, test_size=0.3, dev_size=0.1)
            # 4. Data Preprocessing
            # flatten the images
            X_train = preprocessing(X_train)
            X_test = preprocessing(X_test)
            X_dev = preprocessing(X_dev)
            # HYPER PARAMETER TUNNING

            model_type = "logisticRegression"
            h_param = classifier_param_dict[model_type]
            best_hparams, best_model_path, dev_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, h_param, model_type)
            # Loading best model
            best_model = load(best_model_path)

            train_acc = sum(y_train == best_model.predict(X_train)) / len(y_train)
            test_acc = sum(y_test == best_model.predict(X_test)) / len(y_test)
            # predict_and_eval(best_model, X_test, y_test)
            print("Test Results for model type =", model_type)
            print("test size =", test_size, "dev size =", dev_size, "train size =", 1 - (test_size + dev_size), "train_acc =", train_acc,
                  "test_acc =", test_acc, "dev_acc =", dev_accuracy)

            run_results = {"model name": model_type, "test size": test_size, "dev size": dev_size,
                           "train size": 1 - (test_size + dev_size), "train_acc": train_acc,
                           "test_acc": test_acc, "dev_acc": dev_accuracy}

            # Perform 5-fold cross-validation

            model_files = os.listdir("models")

            # Perform 5-fold verification for each model
            for model_file in model_files:
                model_path = os.path.join("models", model_file)
                model_cv = load(model_path)            
                cv_results = []
                for _ in range(5):
                    X_train_cv, X_test, X_dev_cv, y_train_cv, y_test, y_dev_cv = split_train_dev_test(X_train, y_train, test_size=0.4, dev_size=dev_size)
                    X_train_cv = preprocessing(X_train_cv)
                    X_dev_cv = preprocessing(X_dev_cv)
                    # best_hparams_cv, _, dev_accuracy_cv = tune_hparams(X_train_cv, y_train_cv, X_dev_cv, y_dev_cv, h_param, model_type)
                    train_acc_cv = sum(y_train_cv == model_cv.predict(X_train_cv)) / len(y_train_cv)
                    cv_results.append(train_acc_cv)

                cv_mean = np.mean(cv_results)
                cv_std = np.std(cv_results)

                print("Cross-validation results (mean ± std):", cv_mean, "±", cv_std)
run_exp()


