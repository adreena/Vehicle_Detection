**Vehicle Detection Project**

The goals / steps of this project are the following:

* Step 1: Loading and splitting images for training and testing
* Step 2: Extracting features from training set images
* Step 3: Training classifier LinearSVM on the extracted features
* Step 4: Calculating accuracy of the model using test set images
* Step 5: Searching for vehicles by sliding window over the image and using trained model
* Step 6: Comparing new detected vehicles with the vehicles from the previous frames and merging bounding-boxes if necessary, using heatmap and boxes' areas and centers
* Step 7: Avoiding false positive bounding boxes to appear on the frames
* Step 6: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Goals:
* Developing a robust pipeline to detect vehicles on the road


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### 1-Data

As it was pointed out in the Tips and Tricks for The Project, dataset images are extracted from video which results in almost similar images in a sequence of frames. Even shuffling and splitting the data in a random manner causes overfitting because images in the training set may be nearly identical to images in the test set. So, I took the first `80%` of the images of each category [GTI_Far, GTI_Left, GTI_MiddleClose, GTI_Right, GTI_extracted, KITTI_extracted] as my training set, and left the `20%` of them for testing the model, this helps keeping time-series images in either training-set or testing-set and not in both to make sure train and test images are sufficiently different from one another But just out of curiosity, I implemented 2 versions for my model which I explain in Model section 3.

To add more data to the current data-set I flipped sample images under the same label to cover more case:

<table style="height: 64px; width: 319px;">
  <tbody>
  <tr style="height: 24px;">
  <td style="width: 141px; text-align: center; height: 24px;">Original Image</td>
  <td style="width: 164px; height: 24px;">Flipped Image</td>
  </tr>
  <tr>
  <td><img src="./document/combined-1.png" width="450" height="200"/></td>
  <td><img src="./document/combined-1.png" width="450" height="200"/></td>
  </tr>
  </tbody>
</table>

### 2- Features

After preparing the sets, I converted images to `YCrCb` color-space and collected 3 different sets of features including:

#### 2.1-Spatial Features

Original images size is (64, 64, 3) and contains good spatial features in each channel by showing how target object looks like, but collecting all features for all channels would generate a lot of features and slow down the classifier, yet resizing them to (32,32,3) keeps alsmot  all of the important spatial features in finding vehicles and reduces the feature-set size significantly:
 
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

#### 2-2.Histogram Featurs

Individual histogram of the color channels is another source of information to help classifier detect structures/edges despite the variety of colors. I collected histogram features of all 3 channels. (code : features.py > color_hist())

### 2-3.Histogram of Oriented Gradients (HOG)

For identifying the shape of an object I used `skimage.hog()` to extract the HOG gradient. After running a few experiments I picked theese parameter for the hog
* orientations: 9 
* pix_per_cell: (8,8), I also tried (16,16) to make model train faster , but it reduced the number of vehicle detections in my pipeline
* cell_per_block:  (2,2) I also tried (1,1) along with the (16,16) as pix_per_cell, but as mentioned in the previousl line it reduced accuracy of vehicle detection.

Here are some example using the `YCrCb` color space (channel 2) and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

<table style="width:100%">
  <tr>
    <td>Car </td>
    <td>Hog Channel 2</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
  <tr>
    <td>Car </td>
    <td>Hog Channel 2</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
  <tr>
    <td>Not Car </td>
    <td>Hog Channel 2</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
  <tr>
    <td>Not Car </td>
    <td>Hog Channel 2</td>
  </tr>
  <tr>
    <td><img src="./document/combined-1.png" width="450" height="200"/></td>
    <td><img src="./document/combined-2.png" width="450" height="200"/></td>
  </tr>
</table>

(code: parameters are located in params.py, and teh feature collector modules are in feature.py)

### 3- Model 

I trained 2 separate models just to see how I can improve vehicle detection:

* `model1.p`, for my first model I gathered features from all of the images and used `train_test_split` to split data randomly into training-set and test-set. I then trained my LinearSVC() using YCrCB color space. Although I could see more bounding-boxes in my heatmaps, I observed a ton of false positives happening in the same wrong spot of the road sequentially! Increasing or decreasing the light in the frames just resulted in more false positives.

* `model2.p`: To overcome this issue, I took the first `80%` of the images of each category [GTI_Far, GTI_Left, GTI_MiddleClose, GTI_Right, GTI_extracted, KITTI_extracted] as my training set, and left the 20% of them for testing the model, this helps keeping time-series images in either training-set or testing-set and not in both to make sure train and test images are sufficiently different from one another. (code : model.py > collect_data()) & (flipping the images code: features.py > process_features())
 
<table style="height: 134px; width: 612px;">
<tbody>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">&nbsp;</td>
   <td style="width: 277px; text-align: center; height: 13px;" colspan="2">&nbsp;model2.p (my good&nbsp;model)</td>
   <td style="width: 265px; text-align: center; height: 13px;" colspan="2">model1.p</td>
   </tr>
   <tr style="height: 59px;">
   <td style="width: 83px; height: 59px;">Samples</td>
   <td style="width: 126px; height: 59px;">
   <p>&nbsp;train_cars: 7302</p>
   <p>&nbsp;test_cars: 1760</p>
   </td>
   <td style="width: 151px; height: 59px;">
   <p>&nbsp;train_not_cars: 7174</p>
   <p>&nbsp;test_not_cars: &nbsp;1794</p>
   </td>
   <td style="width: 101px; height: 59px;">&nbsp;cars: 8792</td>
   <td style="width: 164px; height: 59px;">&nbsp;not_cars: 8968</td>
   </tr>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">Train set</td>
   <td style="width: 126px; height: 13px;">&nbsp;X_train: 28412</td>
   <td style="width: 151px; height: 13px;">&nbsp;y_train : 28412</td>
   <td style="width: 101px; height: 13px;">&nbsp;X_train:14208</td>
   <td style="width: 164px; height: 13px;">&nbsp; y_train:14208</td>
   </tr>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">Test set</td>
   <td style="width: 126px; height: 13px;">&nbsp;X_test: 3554</td>
   <td style="width: 151px; height: 13px;">&nbsp;y_test : 3554</td>
   <td style="width: 101px; height: 13px;">&nbsp;X_test: 3552</td>
   <td style="width: 164px; height: 13px;">&nbsp;y_test:3552</td>
   </tr>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">&nbsp;Accuracy</td>
   <td style="width: 126px; height: 13px;" colspan="2">0.9834&nbsp;&nbsp;</td>
   <td style="width: 101px; height: 13px;" colspan="2">&nbsp; 0.9907</td>
   </tr>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">Training time</td>
   <td style="width: 126px; height: 13px;" colspan="2">&nbsp;10.95 sec</td>
   <td style="width: 101px; height: 13px;" colspan="2">6.26 sec</td>
   </tr>
   <tr style="height: 13px;">
   <td style="width: 83px; height: 13px;">Prediction Time for 100 label</td>
   <td style="width: 126px; height: 13px;" colspan="2">0.0009 sec</td>
   <td style="width: 101px; height: 13px;" colspan="2">0.00005 sec</td>
   </tr>
   </tbody>
</table>

### 4- Pipeline 
#### 4-1 Sliding Window Search

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

