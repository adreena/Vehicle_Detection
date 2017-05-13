from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from features import gather_features, gather_features2
from sklearn import svm
from params import *
import pickle
from sklearn.externals import joblib


def collect_data():
    # Read in cars and notcars
    car_images ={}
    car_images['far'] = glob.glob('vehicles/GTI_Far/*.png')
    car_images['left'] = glob.glob('vehicles/GTI_Left/*.png')
    car_images['middle_close'] = glob.glob('vehicles/GTI_MiddleClose/*.png')
    car_images['right'] = glob.glob('vehicles/GTI_Right/*.png')
    car_images['extracted'] = glob.glob('vehicles/KITTI_extracted/*.png')

    not_car_images={}
    not_car_images['gti'] = glob.glob('non-vehicles/GTI/*.png')
    not_car_images['extra'] = glob.glob('non-vehicles/Extras/*.png')

    train_cars = []
    test_cars = []
    train_not_cars = []
    test_not_cars = []
    split_percent = 0.8
    for key,image_list in car_images.items():
        split = int(len(image_list)*split_percent)
        train_cars.extend(image_list[:split])
        test_cars.extend(image_list[split:])
    for key,image_list in not_car_images.items():
        split = int(len(image_list)*split_percent)
        train_not_cars.extend(image_list[:split])
        test_not_cars.extend(image_list[split:])

    print(' ---------------------------------------------')
    print('|Train samples | cars: {} | not_cars: {}  |'.format(len(train_cars), len(train_not_cars)))
    print('|---------------------------------------------|')
    print('|Test samples  | cars: {} | not_cars: {}  |'.format(len(test_cars), len(test_not_cars)))
    print(' ---------------------------------------------')

    return train_cars, train_not_cars, test_cars, test_not_cars

def model():
    print('Reading dataset files ...')
    train_cars, train_not_cars, test_cars, test_not_cars = collect_data()
    print(u'\u2713', 'Done!\n')

    print('Gathering features ...')
    X_train, X_test, y_train, y_test, X_scaler = gather_features2(train_cars,test_cars,train_not_cars,test_not_cars)
    print(u'\u2713', 'Done!\n')

    print(' -----------------------------------------')
    print('|Train set | X_train:{} y_train:{}  |'.format(len(X_train), len(y_train)))
    print('|-----------------------------------------|')
    print('|Test set  | X_test: {} y_test:{}     |'.format(len(X_test), len(y_test)))
    print(' -----------------------------------------')


    joblib.dump(X_scaler, 'Xscaler2.pkl')
    print(u'\u2713', 'Saved XScaler!\n')

    print('shuffled training data')
    X_train, y_train = shuffle(X_train, y_train)
    print(u'\u2713', 'Done!\n')

    print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')

    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    # Check the training time for the SVC
    t=time.time()
    svc = LinearSVC()
    print('Training started...')
    svc.fit(X_train, y_train)

    t2 = time.time()
    print('Took {}(sec) to train SVC...'.format(round(t2-t, 2)))
    # Check the score of the SVC
    t=time.time()
    acc = round(svc.score(X_test, y_test), 4)
    t2 = time.time()
    print('Test Accuracy of SVC = ',acc, ' Took {}(sec):'.format(round(t2-t, 2)) )

    # Check the prediction time for 100 single sample
    t=time.time()
    n_predict = 100
    svc.predict(X_test[0:n_predict])
    t2 = time.time()
    print('Took {} Seconds to predict {} samples'.format(round(t2-t, 4), n_predict))

    filename = 'model2.p'
    pickle.dump(svc, open(filename, 'wb'))
    print(u'\u2713' , 'Model Saved !')


if __name__ == '__main__':
    model()
