**Vehicle Detection Project**

The goals / steps of this project are the following:

* Step 1: Loading and splitting images for training and testing
* Step 2: Extracting features from training set images
* Step 3: Training classifier LinearSVM on the extracted features
* Step 4: Calculating accuracy of the model using test set images
* Step 5: Searching for vehicles by sliding window over the image and using trained model
* Step 6: Comparing new detected vehicles with the vehicles from the previous frames and merging bounding-boxes if necessary, using heatmap and boxes' areas
* Step 7: Avoiding false positive bounding boxes to appear on the frames
* Step 6: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Goals:
* Developing a robust pipeline to detect vehicles on the road


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Data

As it was pointed out in the Tips and Tricks for The Project, dataset images are extracted from video which results in almost identical images in a sequence of frames. Even shuffling and splitting the data in a random manner causes overfitting because images in the training set may be nearly identical to images in the test set. To overcome this issue, I take the first 80% of the images of each category and put them in training set and leave the 20% of them for testing the model, this helps keeping time-series images in either training-set or testing-set and not in both to make sure train and test images are sufficiently different from one another. 

(code : model.py > collect_data())

 ---------------------------------------------
|Train samples | cars: 7032 | not_cars: 7174  |
|---------------------------------------------|
|Test samples  | cars: 1760 | not_cars: 1794  |
 ---------------------------------------------
 
 To generate more data, I added flipped images to each set under the same label, here is an example of a car-image an its flipped version 
 
 (code : features.py > process_features() > 110-118):
 
 <table style="width:100%">
  <tr>
    <td>Original</td>
    <td>Flipped</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
</table>

Total number of images after adding flipped images (cars+not-cars):
 -----------------------------------------
|Train set | X_train:28412 y_train:28412  |
|-----------------------------------------|
|Test set  | X_test: 3554 y_test:3554     |
 -----------------------------------------


### Features

After preparing the sets, I converted images to `YCrCb` color-space and collected 3 different sets of features including:

### 1.Spatial Features

Original images size is (64, 64, 3) and contains good spatial features in each channel by showing how target object looks like, but collecting all features for all channels would generate a lot of features and slow down the classifier, yet resizing them to (32,32,3) keeps important spatial features in finding vehicles and reduces the feature-set size significantly:
 
 (code : features.py > bin_spatial()):

<table style="width:100%">
  <tr>
    <td>Original(64x64x3)</td>
    <td>Shrinked (32x32x3)</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
</table>

### 2.Histogram Featurs

Individual histogram of the color channels is another source of information to help classifier detect structures/edges despite the variety of colors. I collected histogram features of all 3 channels : WHY 32???
(code : features.py > color_hist())
<table style="width:100%">
  <tr>
    <td>Original(64x64x3)</td>
    <td>Channel 1</td>
    <td>Channel 2</td>
    <td>Channel 3</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
     <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
</table>


### 3.Histogram of Oriented Gradients (HOG)

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

