
import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.utils import np_utils



sys.path.append(os.getcwd())
from classification import models
from classification import vizualization
#from preprocessing.create_part_category_list import create_part_category_list


def build_train_test_set(dataset_path, test_size=0.2, df=None, category_dict=None, category_thres=30):
    """
         Builds dataset and splits in train and test set.
         Variant 1: Every part is one class
         Variant 2: Every category is one class (use df, category_dict)

         Args:
             dataset_path: Each subdirectory folder holds the class name and includes images in .jpg format
             test_size: size of the test set

         Returns:
             x_train: ndarray trainings set with shape (instances train, 224, 224, 3)
             x_test: ndarray test set with shape (instances test , 224, 224, 3)
             y_train: ndarray labels from 1 to num_classes for training with shape
             y_test ndarray labels from 1 to num_classes for test
             num_classes: total number of unique classes
             classes: List of class labels
     """


    # collect all filenames
    file_name_list = []
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):
        for directory in dirnames:
            #print(os.path.join(dirpath, directory))
            for root, directories, filenames in os.walk(os.path.join(dirpath, directory)):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if filename.endswith('.jpg'):
                        file_name_list.append(filepath)

    # build train and testset
    xtotal = np.zeros((len(file_name_list),224,224,3), dtype=np.float16)
    ytotal = []
    for i, file_name in enumerate(file_name_list):
        # print(file_name)
        #image = load_img(file_name, color_mode="grayscale", target_size=(224, 224)) # uncomment when using vgg
        image = load_img(file_name, target_size=(224, 224))
        image = img_to_array(image)
        xtotal[i] = image
        splitted = file_name.split("/")
        label = splitted[len(splitted) - 2]
        if df != None:
            # remove unnecessary categories (quantitatively less than category_thres classes)
            try:
                row = df.loc[df['id'] == label]
                if category_dict[row.category.values[0]] >= category_thres:
                    ytotal.append(label)
                else:
                    ytotal.append('other')
            except KeyError:
                continue
        else:
            ytotal.append(label)

    classes = list(set(ytotal))
    num_classes = len(classes)
    print(classes)

    # convert class labels to integer values
    ytotal = [classes.index(label) for label in ytotal]
    print(xtotal.shape)
    #xtotal = np.array(xtotal, dtype=np.float16)
    ytotal = np.array(ytotal)

    x_train, x_test, y_train, y_test = train_test_split(xtotal,
                                                        ytotal,
                                                        test_size=test_size,
                                                        random_state=43,
                                                        shuffle=True)

    print("size of all train images: {} ".format(x_train.shape))
    print("size of all train labels: {} ".format(y_train.shape))
    return x_train, x_test, y_train, y_test, num_classes, classes


def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

def main(dataset_path, variant, model_path=None, model_save=None):

    if variant == "categories": # category classification
        import pandas as pd
        df = pd.read_csv("results/parts_category_list.csv")
        df = df.sort_values(['category', 'label'])

        # create_part_category_list(dataset_path)
        vizualization.plot_dataset_distribution("results/parts_category_list.csv")

        print("category types: {}".format(df.category.unique()))
        print("number of categories: {}".format(df.category.unique().size))
        print(df.category.value_counts())

        category_dict = df.category.value_counts().to_dict()
        print(category_dict)
        x_train, x_test, y_train, y_test, num_classes, classes = build_train_test_set(dataset_path,
                                                                                      df,
                                                                                      category_dict)

    elif variant == "parts":
        print("Build training and test set...")
        x_train, x_test, y_train, y_test, num_classes, classes = build_train_test_set(dataset_path)


    # convert label to categorical
    y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

    if model_path == None:
        model = models.get_custom_VGG19(num_classes)
        #model = models.get_alex_net_model(num_classes, (224, 224, 1))
    else:
        model = load_model(model_path)



    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', 
                optimizer=opt, 
                metrics=['accuracy'])

    if model_path == None:
        history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=20,
                    validation_data=[x_test, y_test],
                    shuffle=True,
                    callbacks=[ModelCheckpoint("results/model.h5", save_best_only=True)]
                            )

        # plot accuracy and loss
        vizualization.plot_training_history_accuracy(history)
        vizualization.plot_training_history_loss(history)

        # save model
        model.save(model_save)

    y_pred = model.predict(x_test)
    vizualization.plot_confusion_matrix(y_pred, y_test, classes)

    # final evaluation
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    import sys, argparse
    argv = sys.argv[1:]

    # when --help or no args are given
    usage_text =  "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)

    # create arguments
    parser.add_argument(
        "-d", "--dataset_path", dest="input", type=str, required=True,
        help="Input folder for train and test data"
    )

    parser.add_argument(
        "-m", "--model", dest="model", type=str, required=False,
        help="Select existing model"
    )

    parser.add_argument(
        "-s", "--save", dest="model_save", type=str, required=False,
        default="results/model.h5",
        help="Save model to..."
    )

    parser.add_argument(
        "-v", "--variant", dest="variant", type=str, required=True,
        help="Set 'parts' to classify parts and set 'categories' to classify categories"
    )

    args = parser.parse_args(argv)
 
    if not argv:
        parser.print_help()
        sys.exit(-1)
    if not (args.input or args.variant):
        print("Error: Some required arguments missing")
        parser.print_help()
        sys.exit(-1)

    if args.variant != 'parts' and args.variant != 'categories':
        print("Error: Choose a correct variant")
        parser.print_help()
        sys.exit(-1)

    dataset_path = os.path.join(os.getcwd(), args.input)
    if not os.path.isdir(dataset_path):
        print("Dataset input path is not correct")
        sys.exit(-1)

    main(dataset_path, args.variant, args.model, args.model_save)