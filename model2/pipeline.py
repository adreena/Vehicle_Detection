import sys
import os
sys.path.insert(0, os.path.realpath('./'))
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
from src.features import *
from sklearn import svm
from src.params import *
import pickle
from src.Vehicle import Vehicle
import pickle
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from src.utils import *

def pipeline(image, file_name=''):
    global vehicles
    global sequence
    global svc
    global X_scaler

    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    im = np.copy(draw_image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    im = im.astype(np.float32)/255

    ystart = 400
    ystop = 656

    scale = 1.5
    #'RGB2YCrCb' , 'BGR2YCrCb' , 'LUV', 'HSV', 'HLS', 'YUV':
    color_space = 'RGB2YCrCb'
    bboxes = find_cars(im, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
    heat = add_heat(heat,bboxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img,centers, vehicles = draw_labeled_bboxes(np.copy(image), labels, vehicles,sequence)
    i =0
    # font = cv2.FONT_HERSHEY_SIMPLEX
    if sequence is True:
        for vehicle in vehicles:
            # cv2.putText(draw_img,'{} : {} ,{}'.format(vehicle.center,vehicle.count_appeared,vehicle.old_count_appeared),(850,50+i), font, 1,(255,255,255),2)
            # i+=50
            vehicle.old_count_appeared=vehicle.count_appeared

    if sequence is False:
        fig = plt.figure()
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.savefig('model2/output_images/{}.jpg'.format(file_name))
        print(u'\u2713', 'Saved heatmap!\n')

    return draw_img


if __name__ == "__main__":
    vehicles = []
    svc = pickle.load( open( "model2/model.p", "rb" ) )
    X_scaler = joblib.load('model2/Xscaler.pkl')
    print(u'\u2713', 'Loaded Model & X_scaler!\n')

    sequence = False
    print('.................test3...................')
    o1= pipeline(mpimg.imread('test_images/test3.jpg'), file_name='test3')
    mpimg.imsave("model2/output_images/output_test3.jpg", o1)
    print('\n')
    print('.................test2...................')
    o1=pipeline(mpimg.imread('test_images/test2.jpg'), file_name='test2')
    mpimg.imsave("model2/output_images/output_test2.jpg", o1)
    print('\n')
    print('.................test4...................')
    o1=pipeline(mpimg.imread('test_images/test4.jpg'), file_name='test4')
    mpimg.imsave("model2/output_images/output_test4.jpg", o1)
    print('\n')
    print('.................test5...................')
    o1=pipeline(mpimg.imread('test_images/test5.jpg'), file_name='test5')
    mpimg.imsave("model2/output_images/output_test5.jpg", o1)
    print('\n')
    print('.................test6...................')
    o1=pipeline(mpimg.imread('test_images/test6.jpg'), file_name='test6')
    mpimg.imsave("model2/output_images/output_test6.jpg", o1)


    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    vehicles = []
    sequence = True
    video_output = 'model2/project_output.mp4'
    clip = VideoFileClip('project_video.mp4')
    project_clip = clip.fl_image(pipeline)
    project_clip.write_videofile(video_output, audio=False)
