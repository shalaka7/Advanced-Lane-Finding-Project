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
[image4]: ./output_images/test1.jpg "Road Transformed"
[image5]: ./output_images/preprocessImage_1.jpg "Binary Example"
[image6]: ./output_images/warped_1.jpg "perspective transform"
[image7]: ./output_images/v1_histogram_1.jpg "Warp Example"
[image8]: ./output_images/histogram_cr_1.jpg "Fit the lines "
[image9]: ./output_images/tracked_1.jpg "Output"
[video1]: ./project_video_output.mp4 " output Video"



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
(thresholding steps are [here](./main_image.py#L76-L136) 
Here's an example of my output for this step.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which Warp an image using the perspective 
transform.The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) 
points.In the code perspective transform [here](./main_image.py#L169-L183)
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

Then I found my lane lines with a 2nd order polynomial by using [function](./main_image.py#L299-L322) like this

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[I did this  in my code in `main_image.py`](./main_image.py#L331-L376)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[I implemented this step in my code in `main_image.py`](./main_image.py#L383-L402)
Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video

 And it works properly !!
 Here's a [link to project video result](./project_video_output.mp4)




### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  
##Where will your pipeline likely fail?  What could you do to make it more robust?

problems that I faced :
1 : The lanes lines in the challenge and harder challenge videos were extremely difficult to detect. 
They were either too bright or too dull. This prompted me to have R & G channel thresholding and L channel thresholding
2 : Bad Frames, The challenge video has a section where the car goes underneath a tunnel and no lanes are detected
3 : The pipeline seems to fail for the harder challenge video. This video has sharper turns and at very short intervals.
4 : Shadows cast by the lane divider ,Lanes lines change color these also create a problems so we have to focus on them .
 
To avoid this :
1 : Take a better perspective transform: choose a smaller section to take the transform since this video has sharper turns 
and the lenght of a lane is shorter than the previous videos.
2: Average over a smaller number of frames so detection of lanes changes quite fast.
