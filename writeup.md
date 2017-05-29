# Advanced Lane Finding Project

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

[undistort]: ./output_images/undistort.png "Undistorted"
[binary]: ./output_images/binary_out.png "Road Transformed"
[out1]: ./output_images/out1.png "Warp Example"
[out2]: ./output_images/out2.png "Binary Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be found in the method called `calibrate_camera()`, between the lines 34 and 57 in the `lanes_detect.py` file.

You can also find the methos `test_calibration_camera` which allow you to see some images before and after the calibration.

Firstly, I load the images of the chessboard which I know it has 6x9 corners. Then I use findChessboardCorners method from the cv2 library so automatically I can find the coordinates of every corner.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistort]

I will save them in pickle files so I don't have to calibrate everytime the camera.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 74 to 106 in `lanes_Detect.py` file which is the method `threshold_image`.  Here's an example of my output for this step.  (note: this binary output is after thresholding)

![alt text][binary_out]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in a method called `perspective_transform` from the line 143 to 157.

The origin position are based in the slope of the lanes with respect the car, which I observed it had a value of `slope = 0.65`.  
The values for source and origin are the next:

src_points = np.float32([[250,base], 
                        [int(((height-base)/slope)+244), height], 
                        [int(((height-base)/(slope*-1))+1036),height], 
                        [1030,base]])
dst_points = np.float32([[middle-space_lines,bottom], 
                        [middle-space_lines,0], 
                        [middle+space_lines,0], 
                        [middle+space_lines,bottom]])

You can see in the last image how the lanes are paralel after the transformation (you have to remember the last image has the transformation of perpective applied)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There is two methos that are used to calculate the lanes:
First one is `find_lanes`. You can find this method between 159 and 225 lines.
This one uses a region of interest to clear the part of the screen where we won't find a lane. Then I used a sliding window with the histogram of the thresholded image so I could find the pixels that represent the lanes.

In the next image, the bottom figure represents the transformed image with 2 black lines with the fitted lane and behind it in red and blue you can see the pixels that are identified as lanes.

![alt text][out1]

The second method to find lanes called `find_lines_margin` uses the same method to get the pixels and fit a polynomial but instead of using a region of interest premade, it uses a margin around the last polynomial we had from the last frame so the detection can be faster and more robust.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 252 through 262 in my code in the method `get_lines_curvature`. To find the curvature of the road you simply has to count how many pixels are in a know distance. In our case the distance between two lanes so we can transform the sizes from pixel space to real space with proper measurement.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here you can see three images where the first one has a green part which represents the polygon contained between the two lanes detected where we can clearly see that it was properly identified.

![alt text][out2]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the first problems that you have to face once you want to transform the perspective is to find the proper destination points. That is because if you take to much data from further distances, the resolution of the top of the image will be really low due to the amplification of this part, but if you cut to much you can't see really far away to detect curves properly. 

Another problem you face is jumps in the road and momentaneum changes in the conditions as light or shadows. To solve this problem it works really fine to ease the polygons averaging in time the output.

This pipeline will fail with terrain that is too brilliant because it detects saturated pixels and that doesn't help in the robustness of the pipeline. 
