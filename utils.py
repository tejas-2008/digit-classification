
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_digits():
    digits = datasets.load_digits()
    x = digits.images 
    y = digits.target
    return x, y
def preprocessing(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_train_dev_test(x, y, test_size, random_state = 1):
    x_train, xtest, y_train, ytest = train_test_split(x, y, test_size=test_size, shuffle=False)
    x_test, x_dev, y_test, y_dev = train_test_split(xtest, ytest, test_size=0.3, shuffle=False)
    return x_train, x_test, x_dev, y_train, y_test, y_dev

def train_model(x, y, model_prams, model_type = "svm"):
    if model_type == "svm":
        clf = svm.SVC
    model = clf(**model_prams)
    model.fit(x, y)
    return model

def predict_and_eval(model, x_test, y_test):
    predicted = model.predict(x_test)
    
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.


    # 8. Evaluation
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    ###############################################################################
    # If the results from evaluating a classifier are stored in the form of a
    # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    # as follows:


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

