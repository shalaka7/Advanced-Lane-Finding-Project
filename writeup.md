## Writeup Template


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners_found4.jpg/ "corner detection"
[image2]: ./output_images/calibration3.jpg/ "original image"
[image3]: ./output_images/calibration3_undistort.jpg/ "undistort image"
[image4]: ./output_images/test3.jpg "Road Transformed"
[image5]: ./output_images/tracked2_binary_result.jpg "Binary Example"
[image6]: ./output_images/bird_eye2.jpg "perspective transform"
[image7]: ./output_images/pathlines2.jpg "Warp Example"
[image8]: ./output_images/drawlines2.jpg "Fit the lines "
[image9]: ./output_images/tracked2.jpg "Output"
[video1]: ./output1_tracked.mp4 " output Video"
[video2]: ./output2_tracked.mp4 " output of challenge Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients.

The camera matrix computation and distortion coefficient is in the camera_calibration.py.In this code we prepare for 
objpoints and imgpoints. In objpoints there will be (x,y,z)these three coordinates of chessboard.but here chessboard
is fixed so that (x,y) plane at z=0this will helpful because it will same for all chessboard so that we  detect all 
chessboard corner.and imgpoint will be appended with the (x,y)positionof each corner of image plane with each sucessfully
chessboard detection.I then used the output objpoints and imgpoints to compute the camera calibration and distortion
coefficients using the cv2.calibrateCamera() function. then applied this distortion correction to the test
image using the cv2.undistort() function and obtained this result:


![alt text][image1]  
![alt text][image2]
![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image .Here I use gradient throsholding,
magnitude thresholding,direction thresholding.and combined all three thresholding .Applying sobel is to identify pixels 
where the gradient of an image falls within a specified threshold range. Magnitude of gradient is to apply a threshold
to the overall magnitude of the gradient, in both x and y.after I used that image for color thresholding
(thresholding steps are [here](./main_image.py#L73-L130) ) 
Here's an example of my output for this step.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which Warp an image using the perspective 
transform.The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) 
points.In the code perspective transform (line 164 to 171)
I chose the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 4820     | 0, 0          | 
| 810, 482      | 1280, 0       |
| 1250, 720     | 1250, 720     |
| 40, 720       | 40, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
 and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I found my lane lines with a 2nd order polynomial by using [function](./main_image.py#L222-L230) kinda like this

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[I did this  in my code in `main_image.py`](./main_image.py#L245-L249)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[I implemented this step in my code in `main_image.py`](./main_image.py#L250-L267)
Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video

 And it works properly !!
 Here's a [link to project video result](./output1_tracked.mp4)
 Here's a [link to challenging video result](./output2_tracked.mp4)



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  
##Where will your pipeline likely fail?  What could you do to make it more robust?

I failed when sharp turn occurs it couldnot follow the path properly.Here I'll talk about the approach I took, 
what techniques I used, what worked and why, where the pipeline might fail and 
how I might improve it if I were going to pursue this project further.  
