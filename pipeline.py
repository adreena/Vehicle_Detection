
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
from features import *
from sklearn import svm
from params import *
import pickle
from Vehicle import Vehicle
import pickle
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from utils import *

def pipeline(image):
    global vehicles
    global sequence
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    im = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    im = im.astype(np.float32)/255

    ystart = 400
    ystop = 656

    scale = 1.5
    #'RGB2YCrCb' , 'BGR2YCrCb' , 'LUV', 'HSV', 'HLS', 'YUV':
    color_space = 'RGB2YCrCb'
    X_scaler = joblib.load('Xscaler.pkl')
    # print(u'\u2713', 'Loaded XScaler!\n')
    svc = pickle.load(open('vehicle_detector_model.sav', 'rb'))
    # print(u'\u2713', 'Loaded Model!\n')


    bboxes = find_cars(im, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
    heat = add_heat(heat,bboxes)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img,centers, vehicles = draw_labeled_bboxes(np.copy(image), labels, vehicles, sequence)

    return draw_img


if __name__ == "__main__":
    vehicles = []
    sequence = False
    o1= pipeline(mpimg.imread('test_images/test3.jpg'))
    mpimg.imsave("output_images/output_test3.jpg", o1)
    print('.................test1...................')
    o1=pipeline(mpimg.imread('test_images/test1.jpg'))
    mpimg.imsave("output_images/output_test1.jpg", o1)
    print('.................test2...................')
    o1=pipeline(mpimg.imread('test_images/test4.jpg'))
    mpimg.imsave("output_images/output_test4.jpg", o1)
    print('.................test3...................')
    o1=pipeline(mpimg.imread('test_images/test5.jpg'))
    mpimg.imsave("output_images/output_test5.jpg", o1)
    print('.................test4...................')
    o1=pipeline(mpimg.imread('test_images/test6.jpg'))
    mpimg.imsave("output_images/output_test6.jpg", o1)
    print('.................test5...................')

    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    vehicles = []
    sequence = True
    video_output = 'project_output.mp4'
    clip = VideoFileClip('project_video.mp4')
    project_clip = clip.fl_image(pipeline)
    project_clip.write_videofile(video_output, audio=False)
