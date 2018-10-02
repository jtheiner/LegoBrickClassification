
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras import Model, Input
from keras.layers import Flatten, Dense

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import Callback

def plot_training_history_accuracy(hist):
    """
    Plots the training history of a keras model using matplotlib and saves as image to training_process_acc.png.

    Args:
        hist: The Keras history from model.fit()

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
    plt.savefig(os.path.join("results/","training_process_acc.png"))

def plot_training_history_loss(hist):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join("results/","training_process_loss.png"))



def build_train_test_set(dataset_path, test_set=0.2):
    """
        Builds dataset and splits
        in train and test set.

        Args:
            dataset_path: Each subdirectory folder holds the class name and includes images in .jpg format
            test_set: size of the test set

        Returns:
            x_train: ndarray trainings set with shape (instances train, 224, 224, 3) 
            x_test: ndarray test set with shape (instances test , 224, 224, 3)
            y_train: ndarray labels from 1 to num_classes for training with shape
            y_test ndarray labels from 1 to num_classes for test
            num_classes: total number of unique classes   
    """

    classes_dict = {}
    index = 0
    # collect all filenames / class files
    file_name_list = []  
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for directory in dirnames:
            print(os.path.join(dirpath, directory))
            splitted = os.path.join(dirpath, directory).split("/")
            label = splitted[len(splitted) - 1]
            classes_dict[label] = index
            index += 1
            for root, directories, filenames in os.walk(os.path.join(dirpath, directory)):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if filename.endswith('.jpg'):
                        file_name_list.append(filepath)


    print("number of classes {}".format(len(classes_dict)))
    num_classes = len(classes_dict)
    print(classes_dict)

    # build train and testset
    xtotal = []
    ytotal = []
    for file_name in file_name_list:
        print(file_name)
        #image = load_img(file_name, color_mode='grayscale' ,target_size=(224, 224))
        image = load_img(file_name, target_size=(224, 224))
        image = img_to_array(image)
        xtotal.append(image)
        splitted = file_name.split("/")
        label = splitted[len(splitted) - 2] 
        ytotal.append(classes_dict[label])

    x_train, x_test, y_train, y_test = train_test_split(np.array(xtotal), 
                                                        np.array(ytotal), 
                                                        test_size=test_set, 
                                                        random_state=42, 
                                                        shuffle=True)

    print("size of all train images: {} ".format(x_train.shape))
    print("size of all train labels: {} ".format(y_train.shape))
    return x_train, x_test, y_train, y_test, num_classes


def get_custom_VGG16(num_classes):
    # fine tune a VGG16
    from keras.applications.vgg16 import VGG16
    vgg16 = VGG16(weights='imagenet', include_top=True)
    #vgg16.summary()

    # customize last layers
    x = Dense(256, activation='relu', name='fc_1')(vgg16.layers[-4].output)
    x = Dense(64, activation='relu', name='fc_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)

    #print("custom model architecture")
    #print(vgg16.input.shape)
    model = Model(inputs=[vgg16.input], outputs=[predictions])
    #model.summary()
    return model

def get_custom_VGG19(num_classes):
    # fine tune a VGG19
    from keras.applications.vgg19 import VGG19
    vgg19 = VGG19(weights='imagenet', include_top=True)
    # customize last layers
    x = Dense(256, activation='sigmoid', name='fc_1')(vgg19.layers[-4].output)
    x = Dense(64, activation='sigmoid', name='fc_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)
    model = Model(inputs=[vgg19.input], outputs=[predictions])
    return model


if __name__ == "__main__":

    x_train, x_test, y_train, y_test, num_classes = build_train_test_set(dataset_path="../dataset/", test_set=0.2)

    # convert label to categorical
    y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=num_classes)


    #model = get_custom_VGG16(num_classes)
    model = get_custom_VGG19(num_classes)

    # freeze the first 8 layers 
    for layer in model.layers[:8]:
        layer.trainable = False

    opt = sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', 
                optimizer=opt, 
                metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                batch_size=16,
                epochs=20,
                validation_data=[x_test, y_test],
                shuffle=True)

    # save model
    #model.save('results/vgg19finetuned.h5')

    # final evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # plot accuracy and loss
    plot_training_history_accuracy(history)
    plot_training_history_loss(history)
