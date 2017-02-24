# Vehicle-Detection-Tracking 
Udacity Self-Driving Car Engineer Class Project, due on Feb 27th, 2017

**Vehicle Detection Project**

The goals / steps of this project are the following:

In this project, the goal is to write a software pipeline to detect vehicles in a provided video. 

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

First of all, I break down the project into small sections, and each section is independent, we can deep dive into it later, but it also follow the process flow:
* Calibarate the Camera
* Experience the behivours of each technique
* Combin some techniques together
* Builded Classifers (HOG and color, and Neural Network)
* Random Slide Windows
* Combin different Classifers
* Increase Processing Speed FPS
* Markup the bounding box
* Pipeline process for the video

### Calibarate the Camera
Since this project is the following project after project 4, and it is not focus on this task, so I just carry over the method and files from project 4. I resized the video frame from 1280x720 to 720x405. I hope can increase some process speed. Also, the calibarate matrix and resolution has to match.  
```
# Reload the camera calibration matrix and distortion conefficients 
scale = 720/1280 # Scale the 1280 image down to 720 image
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/720x540_cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
```
The calibration matrix and distortion comefficient are reloaded from pickle file. 
I builded a class process that include the openCV cv2.undistort as a function. 
```
def undist(self):
        self.undist = cv2.undistort(self.resize(), self.mtx, self.dist, None, self.mtx)
        return self.undist
```
<p align="center">
 <img src="./output_images/camera_calibrication.png" width="720">
</p>

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code in the Jupyter notebook `Building_classifier.ipynb` will follow the order of this writeup. I will explain each code cell and the results.   

First, I read in a test image, cutout the black car and the white car. Then apply
```
# Read in test image
image = mpimg.imread('./test_images/test1.jpg')
cutout = image[400:500, 810:950]    # Black Car 
#cutout = image[400:510, 1050:1270]   # White Car

bins_range = (0,255)
red= cutout[:,:,0]
rhist = np.histogram(red, bins = 32, range=bins_range, normed = False )
bin_edges = rhist[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
```
<p align="center">
 <img src="./output_images/black_car_color_hist.png" width="720">
</p>
<p align="center">
 <img src="./output_images/white_car_color_hist.png" width="720">
</p>
We can see the two histograms from white car and black car appear totally different, therefore, these are good features to recognize these two different color of cars. 

I then explored different color spaces by following Udacity lessens code:
```
# apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
```
<p align="center">
 <img src="./output_images/white_car_3d_color_h_space.png" width="720">
</p>

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
```
# Read in car and non-car images
cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')
```
The 3d plot may not show some meaningful information, but the following YCrCb color space histograms will be good enough to separe car or not car. This code does not include the HOG feature yet. 
```
car_features_YCrCb = extract_features(cars, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
notcar_features_YCrCb = extract_features(notcars, cspace='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
```
<p align="center">
 <img src="./output_images/black_car_YCrCb_hist_features.png" width="720">
</p>
<p align="center">
 <img src="./output_images/not_car_YCrCb_hist_features.png" width="720">
</p>
Another importent trick is normalize the features. As above pictures shown, the Raw feature only show you three bars, other bars values are too weak to show up. I am using StandardScaler().Fit(X) and transform() function as per the Udacity lessons the scale up all features samples from cars and notcars datasets.
```
# Create an array stack of feature vectors
    X = np.vstack((car_features_RGB, notcar_features_RGB)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
```

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). HOG feature is an expensive function. It tooks:
```
37.76 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 1764
11.68 Seconds to train SVC...
Test Accuracy of SVC =  0.9391
My SVC predicts:  [ 1.  0.  0.  0.  1.  1.  1.  0.  1.  0.]
For these 10 labels:  [ 1.  0.  0.  0.  1.  1.  1.  0.  1.  0.]
0.00405 Seconds to predict 10 labels with SVC
```
For the given dataset, 8000+ cars images and 8000+ notcar images, the accuracy can reach 93.91%. I am looking for higher accuracy. So how about add all color histograms and HOG feature together. 
```
spatial_features = bin_spatial(feature_image, size=spatial_size)
file_features.append(spatial_features)

# Apply color_hist()
hist_features = color_hist(feature_image, nbins=hist_bins)
file_features.append(hist_features)

hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
# Append the new feature vector to the features list
file_features.append(hog_features)

features.append(np.concatenate(file_features))
```
This time, it take 3 times longer to extract features and training, but it take same time to predicts a label, and accuracy reached 99.03%, I am happy with this result.    
```
124.46 Seconds to extract HOG features...
Using: 8 orientations 7 pixels per cell and 2 cells per block
Feature vector length: 9312
38.12 Seconds to train SVC...
Test Accuracy of SVC =  0.9903
My SVC predicts:  [ 1.  0.  1.  0.  0.  0.  0.  0.  1.  0.]
For these 10 labels:  [ 1.  0.  1.  0.  0.  0.  0.  0.  1.  0.]
0.004 Seconds to predict 10 labels with SVC
```

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here is a table shows the using the different color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and  `hog_channel = 0, 1,2,All` and accuracies:

|         | Color Space | orientations | pixels_per_cell | cells_per_block | HOG channel | Accuracy |
| --------|:-------:|:-------:|:-------:|:-------:|:-------:|--------:|--------:|
| one     | RGB | 9 | 8 | 2 | 0   | 0.9750 |
| two     | YUV | 9 | 8 | 2 | 0   | 0.98   |
| three   | YUV | 8 | 7 | 2 | All | 0.99   |
| four    | YCrCb | 8 | 7 | 2 | All | 0.9903 |

I choose the number four with 'YCrCb' with `orientations=8`, `pixels_per_cell=(7, 7)` and `cells_per_block=(2, 2)` and 'All' hog channels simply because it yields highest accuracies 99.03%. It is much higher than simple Nerual Network can achive. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using Udacity provided Vehicle and not-Vehicle dataset. I also make a class Params(): to store all feature settings.  
```
class Params():
    def __init__(
        self, 
        colorspace='YCrBr',
        orient=9,
        pix_per_cell=4, 
        cell_per_block=4, 
        hog_channel='ALL',
        spatial_size=(32, 32),
        hist_bins=32,
        spatial_feat=True,
        hist_feat=True,
        hog_feat=True
    ):
        self.colorspace = colorspace # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient # typically between 6 and 12
        self.pix_per_cell = pix_per_cell # HOG pixels per cell
        self.cell_per_block = cell_per_block # HOG cells per block
        self.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size # Spatial binning dimensions
        self.hist_bins = hist_bins # Number of histogram bins
        self.spatial_feat = spatial_feat # Spatial features on or off
        self.hist_feat = hist_feat # Histogram features on or off
        self.hog_feat = hog_feat  # HOG features on or off
```
The training process includes: 
* Extract both Vehicle and Not_Vehicle dataset with lables
* Scale and Normalize the data
* Split the data into 80% for training, 20% for testing.
* Traing the classifier with training data and label
* Check the prediction
```
car_features = extract_features(cars, color_space=params.colorspace, orient=params.orient, 
                        pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block, 
                        hog_channel=params.hog_channel)
notcar_features = extract_features(notcars, color_space=params.colorspace, orient=params.orient, 
                        pix_per_cell=params.pix_per_cell, cell_per_block=params.cell_per_block, 
                        hog_channel=params.hog_channel)
```
```
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', params.orient,'orientations',params.pix_per_cell,
    'pixels per cell and', params.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
```
```
train_acc = svc.score(X_train,y_train)
test_acc = svc.score(X_test,y_test)
Training accuracy:  1.0
Testing accuracy:  0.990332669889
```
I also build a simple Nerual Network in Keras:
```
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def get_conv(input_shape=(64,64,3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(128,8,8,activation="relu",name="dense1")) # This was Dense(128)
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) # This was Dense(1)
    if filename:
        model.load_weights(filename)        
    return model

model = get_conv()
model.add(Flatten())
model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
```


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

### Credit to 
https://github.com/matthewzimmer/CarND-Vehicle-Detection-P5/blob/master/final-project.ipynb
