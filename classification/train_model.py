
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.applications.vgg16 import VGG16
from keras import Model, Input
from keras.layers import Flatten, Dense

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils


def plot_training_history_accuracy(hist):
    """
    Plots the training history of a keras model using matplotlib and saves as image to training_process.png.

    Args:
        hist: The Keras model

    Returns:
        None: Simple writes result to file
    """
    plt.clf()
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy during training')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("training_process.png")



# read dataset and split in train test
# todo: parse via argument
dataset_path = "../dataset/"

classes_dict = {}
index = 0

file_name_list = []  # collect all filenames / class files
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
for index, file_name in enumerate(file_name_list):
    #image = load_img(file_name, color_mode='grayscale' ,target_size=(224, 224))
    image = load_img(file_name, target_size=(224, 224))
    image = img_to_array(image)
    #image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    xtotal.append(image)
    splitted = os.path.join(dirpath, directory).split("/")
    label = splitted[len(splitted) - 2]
    ytotal.append(classes_dict[label])


x_train, x_test, y_train, y_test = train_test_split(np.array(xtotal), np.array(ytotal), test_size=0.2, random_state=42)

print("size of all train images: {} ".format(x_train.shape))
print("size of all train labels: {} ".format(y_train.shape))

# preprocess input
y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
y_test = np_utils.to_categorical(y_test, num_classes=num_classes)



# fine tune a VGG16
vgg16 = VGG16(weights='imagenet', include_top=True)
vgg16.summary()

# custom last layers
x = Dense(2048, activation='relu', name='fc_1')(vgg16.layers[-4].output)
x = Dense(1024, activation='relu', name='fc_2')(x)
x = Dense(num_classes, activation='softmax', name='pred')(x)

print(vgg16.input.shape)

model = Model(inputs=[vgg16.input], outputs=[x])
model.summary()


# retrain
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 batch_size=32,
                 epochs=3,
                 validation_data=[x_test, y_test],
                 shuffle=True)

plot_training_history_accuracy(hist)

scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)


model.save('results/model/vgg16finetuned.h5')


y_pred = model.predict(x_test[0])
from keras.applications.vgg16 import decode_predictions
label = decode_predictions(y_pred)
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))
