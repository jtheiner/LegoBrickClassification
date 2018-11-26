from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG19


def get_alex_net_model(num_classes, img_size):
    # build an AlexNet

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=img_size,
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_custom_VGG19(num_classes):
    # fine tune a VGG19
    vgg19 = VGG19(weights='imagenet', include_top=True)
    # customize last layers
    x = Dense(512, activation='sigmoid', name='fc_1')(vgg19.layers[-4].output)
    x = Dense(256, activation='sigmoid', name='fc_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='pred')(x)
    model = Model(inputs=[vgg19.input], outputs=[predictions])

    # freeze the first 8 layers
    for layer in model.layers[:8]:
        layer.trainable = False

    return model