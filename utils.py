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

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if color_space == 'BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,color_space):
    boxes= []
    draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
        ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    else:
        ctrans_tosearch = np.copy(img_tosearch)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1, im1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=False)
    hog2, im2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=False)
    hog3,im3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block,vis=True, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return boxes



# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
#     print( heatmap[heatmap > 0] )
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, vehicles,sequence):
    centers={}
    good_vehicles=[]
    new_vehicles = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        center = (np.mean([bbox[0][0], bbox[0][1]]), np.mean([bbox[1][0], bbox[1][1]]))
        centers[car_number]=center
        in_range = False

        if sequence is True:
            if len(vehicles) >0:
                for vehicle in vehicles:
                    if vehicle.covers_range2(bbox) or vehicle.covers_range(center):
                        vehicle.count_appeared +=1

                        top_x = np.int(np.mean([bbox[0][0],vehicle.bbox[0][0],vehicle.bbox[0][0]]))
                        top_y = np.int(np.mean([bbox[0][1], vehicle.bbox[0][1],vehicle.bbox[0][1]]))
                        bottom_x = np.int(np.mean([bbox[1][0],vehicle.bbox[1][0],vehicle.bbox[1][0]]))
                        bottom_y = np.int(np.mean([bbox[1][1], vehicle.bbox[1][1],vehicle.bbox[1][1]]))

                        vehicle.bbox= ((top_x,top_y),(bottom_x,bottom_y))
                        vehicle.center =(np.mean([vehicle.bbox[0][0], vehicle.bbox[0][1]]), np.mean([vehicle.bbox[1][0], vehicle.bbox[1][1]]))

                        in_range= True
                        break

            if in_range is False:
                new_vehicle = Vehicle(center,bbox)
                new_vehicles.append(new_vehicle)
        else:

            new_vehicle = Vehicle(center,bbox)
            new_vehicles.append(new_vehicle)


    #filter bad vehicles except the
    if sequence is True:
        if len(vehicles) > 0:
            for i in range(len(vehicles)):
                if vehicles[i].old_count_appeared ==  vehicles[i].count_appeared:
                    if vehicles[i].not_updating >1:
                        if vehicles[i].count_appeared>1:
                            vehicles[i].count_appeared-=2
                            vehicles[i].old_count_appeared-=2
                            good_vehicles.append(vehicles[i])
                        else:
                            pass
                    elif vehicles[i].not_updating ==1 and vehicles[i].old_count_appeared ==0:
                        pass
                    else:
                        #candidate
                        vehicles[i].not_updating+=1
                        good_vehicles.append(vehicles[i])
                elif vehicles[i].old_count_appeared <  vehicles[i].count_appeared:
                    vehicles[i].not_updating= 0
                    good_vehicles.append(vehicles[i])
                else:
                    pass

    #add
    good_vehicles.extend(new_vehicles)
    for vehicle in good_vehicles:
        if sequence is True:
            if vehicle.count_appeared ==0 and vehicle.old_count_appeared ==0 and vehicle.not_updating>0:
                pass
            else:
                if vehicle.old_count_appeared >-1:
                    cv2.rectangle(img, vehicle.bbox[0],vehicle.bbox[1], (255,0,0), 6)
        else:
            cv2.rectangle(img, vehicle.bbox[0],vehicle.bbox[1], (255,0,0), 6)

    # Return the image
    return img, centers, good_vehicles

def adjust_gamma(image, gamma=1.0):
    '''
    build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values
    '''
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
