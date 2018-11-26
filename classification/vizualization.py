from matplotlib import pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(pred, test, classes):
    """
    Plots a confusion matrix and saves to file (.csv and .eps).
    Saves the image to results/.

    Args:
        pred:   Output prediction vector
        test:   List of binary label vectors for each class e.g. [[0,0,1,..],[0,1,0,..], ...]
        classes: List of class labels

    Returns:
        None:   Simple writes results to file
    """

    
    import matplotlib.ticker as ticker
    from sklearn.metrics import confusion_matrix
    import itertools
    
    fig = plt.figure(figsize=(15,15)) # manually set the figure size depending on matrix size
    # perform argmax to get class index to get a list of indexes [0,1,2,0,2,2,....]
    pred_classes = pred.argmax(axis=-1)
    test_classes = test.argmax(axis=-1)

    
    # convert class index to class label for confusion matrix
    pred_classes = list(map(lambda i: classes[i], pred_classes))
    test_classes = list(map(lambda i: classes[i], test_classes))

    # create the confusion matrix
    cm = confusion_matrix(pred_classes, test_classes, labels=classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    

    fmt = '.2f'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if (format(cm[i, j], fmt) == '0.00'):
            item = '0'
        else:
            item = format(cm[i, j], fmt)
        plt.text(j, i, item,
            horizontalalignment="center",
            color="white" if cm[i, j] < thresh else "black")

    ax = fig.add_subplot(111)
    ax.set_xticklabels([''] + classes, rotation=45)
    ax.set_yticklabels([''] + classes)
    plt.imshow(cm, aspect='auto', cmap=plt.get_cmap("viridis"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # save result to file
    np.savetxt(os.path.join("results/", "confusion_matrix.csv"), cm, delimiter=",")
    plt.savefig(os.path.join("results/", "confusion_matrix.eps"), format='eps')

def plot_dataset_distribution(part_category_csv):
    """

    :param part_category_csv:
    :return:
    """
    import pandas as pd
    df = pd.read_csv(part_category_csv)

    plt.figure(figsize=(35, 5))

    category = df.category
    prob = category.value_counts()
    threshold = 0.00
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    prob['other'] = tail_prob
    prob.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.savefig("results/category_distribution.eps", format='eps')


def plot_training_history_accuracy(history):
    """
    Plots the training accuracy of a keras model using matplotlib and saves as image.

    Args:
        history: The Keras history from model.fit()

    Returns:
        None: Simple writes result to file
    """
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy during training')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join("results/","training_process_acc.eps"), format='eps')

def plot_training_history_loss(history):
    """
    Plots the training loss of a keras model using matplotlib and saves as image.

    Args:
        history: The Keras history from model.fit()

    Returns:
        None: Simple writes result to file
    """
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join("results/","training_process_loss.eps"), format='eps')