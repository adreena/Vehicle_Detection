import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

class Vehicle():
    def __init__(self, center=None, bbox=None):
        self.center=center
        self.count_appeared = 0
        self.old_count_appeared = -1
        self.bbox = bbox
        self.not_updating=0

    def covers_range(self,new_point):
        offset = 25
        if self.center != None and \
            (new_point[0] >= self.center[0]-offset and  new_point[0] <=self.center[0]+offset or \
            new_point[1] >= self.center[1]-offset and  new_point[1] <=self.center[1]+offset):
            return True
        return False

    def covers_range2(self,new_point):
        offset = 0
        if self.bbox != None and \
            (new_point[0][0] >= self.bbox[0][0]-offset and  new_point[0][0] <=self.bbox[1][0]+offset and\
             new_point[1][0] >= self.bbox[0][0]-offset and  new_point[1][0] <=self.bbox[1][0]+offset) and\
            (new_point[0][1] >= self.bbox[0][1]-offset and  new_point[0][1] <=self.bbox[1][1]+offset and\
             new_point[1][1] >= self.bbox[0][1]-offset and  new_point[1][1] <=self.bbox[1][1]+offset):
            return True
        elif(self.bbox[0][0] >= new_point[0][0]-offset and  self.bbox[0][0] <=new_point[1][0]+offset and\
             self.bbox[1][0] >= new_point[0][0]-offset and  self.bbox[1][0] <=new_point[1][0]+offset) and\
            (self.bbox[0][1] >= new_point[0][1]-offset and  self.bbox[0][1]<=new_point[1][1]+offset and\
             self.bbox[1][1] >= new_point[0][1]-offset and  self.bbox[1][1]<=new_point[1][1]+offset):
            return True
        return False
