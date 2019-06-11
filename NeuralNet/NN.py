import os
import imageio
import logging
import sys
import getopt

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn import preprocessing
from sklearn.externals import joblib
from random import shuffle
from PIL import Image

# Image size per dimension
IMG_SIZE = 300

def main(argv):
    TRAIN = True

    # Logging
    logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

    train_dir = None
    model_dir = None
    labels_dir = None
    try:
       opts, args = getopt.getopt(argv, "hi:m:l:",["train_dir=", "model_dir=", "labels_dir="])
    except getopt.GetoptError:
        print('NN.py -i <train_dir> -l <labels_dir> -m <model_dir>')
        sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print('NN.py -i <train_dir> -l <labels_dir> -m <model_dir>')
           sys.exit()
       elif opt in ("-i", "--train_dir"):
           train_dir = arg
       elif opt in ("-m", "--model_dir"):
           model_dir = arg
       elif opt in ("-l", "--labels_dir"):
           labels_dir = arg

    if train_dir is None:
        TRAIN = False
    else:
        if train_dir is None or labels_dir is None:
            logging.info("No directory for training data/labels was provided")
            print('NN.py -i <train__dir> -l <labels_dir> -m <model_dir>')
            sys.exit(2)
    if model_dir is None:
        logging.info("No directory for model save/load was provided")
        print('NN.py -i <train__dir> -l <labels_dir> -m <model_dir>')
        sys.exit(2)

    # Log information data
    logging.info("Image size set to: %d x %d" % (IMG_SIZE, IMG_SIZE))

    #breed_count = renameSet(labels, breeds, breed_set)

    if TRAIN is True:
        # Read the labels from the csv file
        labels = pd.read_csv(labels_dir)
        # Get breed column and create a set to remove duplicates
        breeds = labels['breed']
        breed_set = list(set(breeds))
        labels=labels['id']
        logging.info("Number of breeds in training set: %d" % (len(breed_set)))

        # Preprocess images and create flipped image from original
        x_train, y_train, transform = createData(train_dir)

        # Create model for these data
        logging.info('Creating Neural Net')
        model = neuralNet(x_train, y_train, epochs=10, n_classes=len(breed_set))
        logging.info('Created Neural Net')

        # Evaluate model
        loss, acc = model.evaluate(x_test, y_test, verbose = 1)
        logging.info('Evaluated loss: %.4f' % (loss))
        logging.info('Evaluated accuracy: %.4f\n' % (acc))
        # Save model to file
        saveModel(model, transform, model_dir)
    else:
        # Load model from file
        model, transform = loadModel(model_dir)

    # Predict
    logging.info('Predicting for Ermis...')
    test_data = loadTestImages('Ermis/')
    prediction = model.predict(test_data, verbose=1, steps=1)
    determineResult(prediction, 'Ermis/', transform)
    print('--------------------------------------------------')

    logging.info('Predicting for Fred...')
    test_data = loadTestImages('Fred/')
    prediction = model.predict(test_data, verbose=1, steps=1)
    determineResult(prediction, 'Fred/', transform)
    print('--------------------------------------------------')

    logging.info('Predicting for Jake...')
    test_data = loadTestImages('Jake/')
    prediction = model.predict(test_data, verbose=1, steps=1)
    determineResult(prediction, 'Jake/', transform)
    print('--------------------------------------------------')


"""
Impletents an image prediction neural net for this training set
"""
def neuralNet(x_train, y_train, n_classes = 2, epochs = 1):

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation = 'softmax'))

    # For a multi-class classification problem
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=50, epochs=epochs, verbose=1)

    return model


"""
Preprocess images and labels
Create flipped image to expand training set
Return data and categorical labels
"""
def createData(dir, transform=None):

    data = []
    #for i in range(0, 100):
    #    img = images[i]
    for img in os.listdir(dir):
        label = img.split('-')[0]
        path = os.path.join(dir, img)
        image = Image.open(path)
        img = image.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data.append([np.array(img), label])

        # Basic Data Augmentation - Horizontal Flipping\
        flip_img = np.fliplr(img)
        data.append([flip_img, label])

    shuffle(data)

    # Separate data from labels, add one more dimension in data as it is required from the model
    x_train = []
    y_train = []
    for data in data:
        image = np.expand_dims(data[0], axis=2)
        x_train.append(image)
        y_train.append(data[1])

    # Transform labels to categorical format
    if transform is None:
        transform = preprocessing.LabelEncoder()
        transform.fit(y_train)
    y_train = transform.transform(y_train)
    y_train = to_categorical(y_train)

    return np.array(x_train), y_train, transform


"""
Load images for testing and preprocess them to match the training images
"""
def loadTestImages(dir):

    test_data = []

    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        # Add one dimension in the end to match the expencted format of the model
        img = np.expand_dims(img, axis=2)
        test_data.append(np.array(img))

    return np.array(test_data)


"""
Find the breed with the max probability and report it
"""
def determineResult(predictions, dir, transform):
    images = os.listdir(dir)
    # Number of prdictions should be the same as the number of images
    img = 0
    for prediction in predictions:
        max_probs, max_pos = maxProbabilities(prediction, 5)
        logging.info('Dog in image %s belongs with probability:' % (images[img]))
        for i in range(0, len(max_probs)):
            breed = str(transform.inverse_transform([max_pos[i]]))
            logging.info('%.4f to the breed %s' % (max_probs[i], breed))
        logging.info('\n')
        img += 1


"""
Save model to file in the specified dir
"""
def saveModel(model, transform, dir):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dir + "/model.h5")
    # save transform
    joblib.dump(transform, dir + "/transform.save")
    print("Saved model to disk")


"""
Load model from the specified dir
"""
def loadModel(dir):
    # load json and create model
    json_file = open(dir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(dir + "/model.h5")
    # Load transform
    transform = joblib.load(dir + "/transform.save")
    print("Loaded model from disk")
    return model, transform


"""
Find and return the <size> max probabilities and their positions in the
predictions array
"""
def maxProbabilities(predictions, size):
    max_probs = np.zeros(size)
    max_pos = np.zeros(size, dtype=int)
    i = 0
    for prediction in predictions:
        pos = 0
        while pos < size:
            if max_probs[pos] < prediction:
                j = pos
                current_max = prediction
                current_i = i
                while j < size:
                    temp = max_probs[j]
                    temp_i = max_pos[j]
                    max_probs[j] = current_max
                    max_pos[j] = current_i
                    current_max = temp
                    current_i = temp_i
                    j += 1
                pos = size
            pos += 1
        i += 1
    return max_probs, max_pos


"""
Used once to rename the training set according to the breed of each picture
"""
def renameSet(labels, breeds, breed_set, small=True):
    # load images
    images = os.listdir('data/train')

    # Count each breed example
    breed_count = {}
    for breed in breed_set:
        breed_count[breed] = 0

    # Rename each image and copy in the new folder
    add = True
    for image in images:
        imgName = image.split('.')[0]
        label = labels.index[labels==imgName]
        breed = breeds[label].values[0]
        add = (small is not True) or (breed_count[breed] < 5)

        if add is True:
            breed_count[breed] += 1
            path = os.path.join('data/train', image)
            saveName = 'data/labeled_train_small/' + breed + '-' + str(breed_count[breed]) + '.jpg'
            image_data = np.array(Image.open(path))
            imageio.imwrite(saveName, image_data)

    return breed_count


if __name__ == "__main__":
    main(sys.argv[1:])
